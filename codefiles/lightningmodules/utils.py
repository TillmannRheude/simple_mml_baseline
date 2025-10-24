import torch 
import schedulefree
from metann import ProtoModule
from torch.backends.cuda import sdp_kernel
from unittest.mock import patch

import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl 
from lightning.fabric.utilities.throughput import measure_flops
from tqdm import tqdm
import warnings
import time
import numpy as np
from sklearn.cluster import KMeans
import copy
from torch.utils.data import DataLoader

from codefiles.methods.mmpareto.min_norm_solvers import MinNormSolver

from torchmetrics.functional import auroc
from torchmetrics import Metric, MeanMetric
from torchmetrics.utilities import dim_zero_cat

from codefiles.methods.ebr.ebr import Explicit_Basis_Reallocation_Transformer
from codefiles.methods.smil.smil import SMIL
from codefiles.losses.mcr_losses import SupervisedContrastiveLoss, ConditionalEntropyBottleneck
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss

class LightningModuleParent(pl.LightningModule):
    def __init__(
        self,
        manual_opt: bool = False,
        params_ogm: dict = {
            "use_ge": True,
        },
        params_arl: dict = {
            "unimodal_loss_weight": 4.0,
            "temperature": 4.0,
        },
        params_dgl: dict = {
            "unimodal_loss_weight": 1.0,
        },
        params_mcr: dict = {
            "loss_weight_M": 1.0,
            "loss_weight_uni": [1.0, 1.0],
            "d_model": 512,
            "num_classes": 2,
            "num_modalities": 2,
            "is_multilabel": False,
        },
        params_mmpareto: dict = {
            "unimodal_loss_weight": 1.0, 
            "gamma": 0.1
        },
        params_bmml: dict = {
            "alpha": 0.1,
            "q": 5,
            "warmup_epochs": 1,
            "unimodal_loss_weight": 1.0,
            "num_modalities": 2,
        },
        params_gblend: dict = {
            "mode": "offline",
            "lookahead_epochs": 3,
            "update_freq": 1,
            "num_modalities": 2
        },
        params_pdf: dict = {
            "loss_weight": 1.0,
            "unimodal_loss_weight": 1.0,
            "p_true_loss_fn": "l1"
        },
        params_pmr: dict = {
            "epsilon": 0.99,
        },
        params_omib: dict = {
            "warmup_epochs": 1,
        },
        params_smil: dict = {
            "inner_lr": 1e-4,
            "alpha": 0.1,
        },
        params_albef: dict = {
            "alpha": 0.4,
            "distill_temp": 0.1,
        },
        params_avmc: dict = {
            "unimodal_loss_weights": [1.0, 1.0],
            "consistency_loss_weights": [1.0, 1.0],
            "mixup_alpha": 0.2,
            "modalities_to_mix_keys": [0, 1],
            "consistency_loss_type": "mse" # 'mse', 'l1', 'kldiv'
        },
        params_ebr: dict = {
            "interleave_epochs": 0,
        },
        params_simmlm: dict = {
            "mofe_lambda": 0.1,
        },
        **kwargs
    ) -> None: 
        super().__init__()

        self.manual_opt = manual_opt
        if self.manual_opt:
            self.automatic_optimization = False
        self.params_ogm = params_ogm
        self.params_arl = params_arl
        self.params_dgl = params_dgl
        self.params_mcr = params_mcr
        self.params_mmpareto = params_mmpareto
        self.params_bmml = params_bmml
        self.params_gblend = params_gblend
        self.params_pdf = params_pdf
        self.params_pmr = params_pmr
        self.params_omib = params_omib
        self.params_smil = params_smil
        self.params_albef = params_albef
        self.params_avmc = params_avmc
        self.params_ebr = params_ebr
        self.params_simmlm = params_simmlm

        self.gblend_weights = None
        self.smil_meta_iters_setup = False
        self.summary_logged = False
        self.epoch_start_time = None

        self.save_hyperparameters()

        if self.params_gblend.get("mode") in ["online", "offline"]:
            num_modalities = self.params_gblend.get("num_modalities", 2)
            num_heads = num_modalities + 1
            self.gblend_weights = torch.ones(num_heads) / num_heads
            self.gblend_history = []
            self.gblend_weights_calculated = False
            self.current_train_gblend_losses = []

        # BMML: Balanced Multimodal Learning
        if len(self.params_bmml) > 0:
            num_modalities = self.params_bmml["num_modalities"]
            self.bmml_M_accumulators = {
                "theta": [1e-8] * num_modalities,
                "theta_prime": [1e-8] * num_modalities
            }
            # 'none' or 'm{i}' to indicate which modality to boost
            self.bmml_rebalancing_mode = 'none'
            self.bmml_rebalancing_counter = 0

        # MCR: Multimodal Competition Regularizer
        if len(self.params_mcr) > 1:
            self.l_con = SupervisedContrastiveLoss(
                temperature=self.params_mcr["contrastive_temp"]
            )
            self.l_ceb = ConditionalEntropyBottleneck(
                d_model=params_mcr["d_model"],
                num_classes=params_mcr["num_classes"],
                num_modalities=params_mcr["num_modalities"],
                hidden_dim=self.params_mcr["ceb_reconstruction_head"]["hidden_dim"],
                num_layers=self.params_mcr["ceb_reconstruction_head"]["num_layers"],
                is_multilabel=self.params_mcr["is_multilabel"],
            )

    def setup(self, stage: str) -> None:
        if self.params_gblend.get("mode") == "online":
            num_modalities = self.params_gblend.get("num_modalities", 2)
            self.gblend_train_loss_fused = MeanMetric()
            self.gblend_val_loss_fused = MeanMetric()
            self.gblend_train_losses_uni = nn.ModuleList([MeanMetric() for _ in range(num_modalities)])
            self.gblend_val_losses_uni = nn.ModuleList([MeanMetric() for _ in range(num_modalities)])

    def on_train_epoch_start(self):
        if self.current_epoch == 0 and self.trainer.is_global_zero:
            self.epoch_start_time = time.time()

        if len(self.params_omib) > 0:
            self.model.transformer.set_warmup_phase(self.current_epoch < self.params_omib["warmup_epochs"])

        if len(self.params_pmr) > 0:
            # Update the model's internal epoch counter and update prototypes
            self.model.transformer.current_epoch = self.current_epoch
            self.model.transformer.update_prototypes(
                dataloader=self.trainer.datamodule.train_dataloader(),
                encoders=self.model.encoders,
                get_input_fn=get_input,
                get_target_fn=get_target,
                dataset_name=self.dataset,
                device=self.device,
                epsilon=self.params_pmr.get("epsilon", 0.99)
            )

    def on_fit_start(self):
        if not self.summary_logged and self.trainer.is_global_zero:
            self._log_flops_once()
            self.summary_logged = True

        if self.params_gblend:
            if self.params_gblend.get("mode") != "offline":
                return

            print("G-Blend: Starting offline lookahead weight calculation...")

            def _run_eval_pass(head_idx: int, dataloader):
                """Helper to run a full evaluation pass and return the average loss."""
                self.model.transformer.set_gblend_head_active(head_idx)
                losses = []
                
                for batch in dataloader:
                    batch = self.transfer_batch_to_device(batch, self.device, 0)
                    x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
                    if self.params_mcr and not self.params_mcr["is_multilabel"]:
                        if y.ndim > 1 and y.shape[1] == 1: y = y.squeeze(1)

                    with torch.no_grad():
                        logits = self.model(x, y)["logits"]
                        loss = self.loss(logits.squeeze(), y.squeeze())
                    losses.append(loss.item())
                return sum(losses) / len(losses) if losses else 0.0

            initial_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            num_modalities = self.params_gblend["num_modalities"]
            lookahead_epochs = self.params_gblend["lookahead_epochs"]
            head_indices = list(range(num_modalities)) + [-1] # e.g., [0, 1, -1] for fused
            
            all_losses = []

            for head_idx in head_indices:
                self.model.load_state_dict(initial_model_state)
                self.model.train()
                temp_optimizer = self.configure_optimizers()
                if isinstance(temp_optimizer, schedulefree.AdamWScheduleFree):
                    temp_optimizer.train()
                self.model.transformer.set_gblend_head_active(head_idx)
                
                initial_train_loss = _run_eval_pass(head_idx, self.trainer.datamodule.train_dataloader())
                initial_val_loss = _run_eval_pass(head_idx, self.trainer.datamodule.val_dataloader())

                for _ in range(lookahead_epochs):
                    for batch in self.trainer.datamodule.train_dataloader():
                        batch = self.transfer_batch_to_device(batch, self.device, 0)
                        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
                        if self.params_mcr and not self.params_mcr["is_multilabel"]:
                            if y.ndim > 1 and y.shape[1] == 1: y = y.squeeze(1)

                        logits = self.model(x,y)["logits"]
                        loss = self.loss(logits.squeeze(), y.squeeze())
                        temp_optimizer.zero_grad()
                        self.manual_backward(loss)
                        temp_optimizer.step()

                self.model.eval()
                if isinstance(temp_optimizer, schedulefree.AdamWScheduleFree):
                    temp_optimizer.eval() 
                final_train_loss = _run_eval_pass(head_idx, self.trainer.datamodule.train_dataloader())
                final_val_loss = _run_eval_pass(head_idx, self.trainer.datamodule.val_dataloader())
                
                all_losses.append({
                    'train_start': initial_train_loss, 'val_start': initial_val_loss,
                    'train_end': final_train_loss, 'val_end': final_val_loss,
                })

            raw_weights = []
            for losses in all_losses:
                delta_g = losses['val_start'] - losses['val_end']
                
                delta_o_train = losses['train_end'] - losses['train_start']
                delta_o_val = losses['val_end'] - losses['val_start']
                delta_o = delta_o_train - delta_o_val

                weight = max(delta_g, 1e-8) / (delta_o**2 + 1e-8)
                raw_weights.append(weight)
            
            final_weights = torch.tensor(raw_weights, dtype=torch.float32)
            final_weights = final_weights / (torch.sum(final_weights) + 1e-8)

            self.gblend_weights = final_weights

            self.model.load_state_dict(initial_model_state)
            self.model.transformer.set_gblend_head_active(-1) 

        if self.params_smil:
            self._precompute_smil_priors(self.trainer.datamodule.meta_val_dataloader())

    def _log_flops_once(self):
        """Calculates and logs the model's FLOPs for a single forward pass."""
        try:
            dataloader = self.trainer.datamodule.train_dataloader()
            batch = next(iter(dataloader))
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
            batch_size = x[0].shape[0] if isinstance(x, list) and len(x) > 0 else x.shape[0]

            forward_fn = lambda: self.model(x, y)
            
            # For models like AVMC with internal no_grad blocks, FLOPs calculation can fail.
            # We temporarily patch torch.no_grad to torch.enable_grad to allow the tracer
            # to see all operations. This is safe as no weights are updated here.
            with patch('torch.no_grad', torch.enable_grad):
                total_flops_for_batch = measure_flops(self.model, forward_fn)

            gflops_per_instance = (total_flops_for_batch / batch_size) / 1e9
            
            if self.logger:
                self.logger.log_metrics({"train/GFlops": gflops_per_instance}, step=0)
            print(f"Model GFlops (per instance): {gflops_per_instance:.2f}")

        except Exception as e:
            warnings.warn(f"Could not calculate Flops: {e}")

    def _precompute_smil_priors(self, dataloader):
        """
        Pre-computes the modality priors for SMIL using K-Means clustering.
        This is a core part of the original SMIL paper's methodology.
        """
        print("SMIL: Starting pre-computation of modality priors using K-Means...")
        self.model.eval()

        num_modalities = self.model.transformer.num_modalities
        num_priors = self.model.transformer.num_priors
        
        all_embs = [[] for _ in range(num_modalities)]

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="SMIL: Extracting embeddings for K-Means"):
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                x = get_input(self.dataset, batch)
                
                embs, src_mask = self.model(x, return_details=True) 
                present_mask = ~src_mask

                for i in range(num_modalities):
                    mod_i_present_mask = present_mask[:, i]
                    if mod_i_present_mask.any():
                        present_embs = embs[mod_i_present_mask, i, :]
                        all_embs[i].append(present_embs.cpu().numpy())

        new_priors = []
        for i in range(num_modalities):
            if not all_embs[i]:
                warnings.warn(f"SMIL: No present samples found for modality {i}. Using random initialization for priors.")
                priors_i = torch.randn(num_priors, self.model.transformer.modality_priors.shape[-1])
            else:
                mod_embs_np = np.concatenate(all_embs[i], axis=0)
                
                if len(mod_embs_np) < num_priors:
                    warnings.warn(f"SMIL: Not enough samples ({len(mod_embs_np)}) for modality {i} to form {num_priors} clusters. Using random samples as priors.")
                    indices = np.random.choice(len(mod_embs_np), num_priors, replace=True)
                    priors_i_np = mod_embs_np[indices]
                else:
                    kmeans = KMeans(n_clusters=num_priors, n_init=10)
                    kmeans.fit(mod_embs_np)
                    priors_i_np = kmeans.cluster_centers_
                
                priors_i = torch.from_numpy(priors_i_np).float()

            new_priors.append(priors_i)

        final_priors = torch.stack(new_priors).to(self.device)
        self.model.transformer.modality_priors.data = final_priors
        self.model.transformer.modality_priors.requires_grad = False

        print("SMIL: Modality priors pre-computation complete. Priors are now frozen.")
        self.model.train()

    def _compute_and_set_ebr_ranking(self, dataloader):
        """
        Computes the modality similarity ranking required for EBR's inference-time
        substitution strategy (Sec 4.4).
        """
        print("EBR: Computing modality similarity ranking...")
        self.model.eval()

        num_modalities = len(self.model.encoders.encoders)
        similarity_matrix = torch.zeros((num_modalities, num_modalities), device=self.device)
        counts = torch.zeros((num_modalities, num_modalities), device=self.device)

        with torch.no_grad():
            for batch in dataloader:
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                x = get_input(self.dataset, batch)

                g_embs, src_mask = self.model(x, return_details=True) 

                embs_norm = F.normalize(g_embs, p=2, dim=-1)
                batch_similarities = torch.bmm(embs_norm, embs_norm.transpose(1, 2))

                # Create a mask for pairs where both modalities are present
                available_mask = ~src_mask
                available_pairs_mask = torch.bmm(
                    available_mask.unsqueeze(2).float(), 
                    available_mask.unsqueeze(1).float()
                ).bool()

                # Accumulate similarities and counts for present pairs
                similarity_matrix += torch.sum(batch_similarities * available_pairs_mask, dim=0)
                counts += torch.sum(available_pairs_mask, dim=0)
        
        avg_similarity = similarity_matrix / (counts + 1e-8)
        avg_similarity.fill_diagonal_(-torch.inf)
        
        # Sort indices for each modality to get the ranked list of substitutes
        ranking = torch.argsort(avg_similarity, dim=1, descending=True)
        
        final_ranking = []
        for i in range(num_modalities):
            ranked_indices = ranking[i]
            # Filter out the modality's own index to get a list of substitutes
            substitutes = ranked_indices[ranked_indices != i]
            final_ranking.append(substitutes)
        
        final_ranking_tensor = torch.stack(final_ranking)
        self.model.transformer.modality_ranking = final_ranking_tensor.to(self.device)
        print("EBR: Modality ranking computed and set in the model.")

    def _get_metric_for_split(self, split: str):
        """Tries to find a metric object for the given split based on pre-defined patterns."""
        for pattern in ["acc_2_{}", "metric_{}_macro"]:
            metric_name = pattern.format(split)
            metric = getattr(self, metric_name, None)
            if metric:
                return metric
        return None

    def forward(
        self, 
        x: list = [torch.Tensor],
        y: torch.Tensor = None,
    ) -> torch.Tensor:
        current_epoch = self.trainer.current_epoch
        steps_per_epoch = self.trainer.num_training_batches
        return self.model(x, y, epoch=current_epoch, steps_per_epoch=steps_per_epoch)
    
    def shared_step(
        self, 
        batch: tuple,
        set: str = "train",
        convert_logits: str = None
    ) -> torch.tensor:
        if self.manual_opt and self.training:
            opt = self.optimizers()
            opt.zero_grad()
                        
            if self.params_smil:
                # The efficient attention backend does not support second-order gradients
                with sdp_kernel(enable_flash=False, enable_mem_efficient=False):
                    meta_train_batch = batch

                    try:
                        meta_val_batch = next(self.meta_val_loader_iter)
                    except StopIteration:
                        self.meta_val_loader_iter = iter(self.trainer.datamodule.meta_val_dataloader())
                        meta_val_batch = next(self.meta_val_loader_iter)

                    meta_val_batch = self.transfer_batch_to_device(meta_val_batch, self.device, 0)

                    x_train, y_train = get_input(self.dataset, meta_train_batch), get_target(self.dataset, meta_train_batch)
                    x_val, y_val = get_input(self.dataset, meta_val_batch), get_target(self.dataset, meta_val_batch)

                    # --- INNER LOOP ---
                    params = list(self.model.transformer.parameters())
                    
                    embs_train, src_mask_train = self.model(x_train, return_details=True)

                    output_train = self.model.transformer.functional(params, True)(x=embs_train, src_mask=src_mask_train)
                    loss_meta_train = self.loss(output_train["logits"], y_train)

                    trainable_params = [p for p in params if p.requires_grad]
                    grads = torch.autograd.grad(loss_meta_train, trainable_params, create_graph=True, allow_unused=True)
                    
                    params_star = []
                    grad_iter = 0
                    for p in params:
                        if p.requires_grad:
                            grad = grads[grad_iter]
                            if grad is not None:
                                params_star.append(p - self.params_smil["inner_lr"] * grad)
                            else:
                                params_star.append(p)
                            grad_iter += 1
                        else:
                            params_star.append(p)

                    # --- OUTER LOOP ---
                    # The meta_val_batch is guaranteed to be complete. src_mask_val_complete should be all False.
                    embs_val, src_mask_val_complete = self.model(x_val, return_details=True)

                    # Artificially create a missing modality for the "student" path
                    present_mask_val = ~src_mask_val_complete
                    present_indices_val = [torch.where(row)[0] for row in present_mask_val]
                    missing_indices_val = torch.tensor([
                        indices[torch.randint(len(indices), (1,))] if len(indices) > 0 else 0
                        for indices in present_indices_val
                    ], device=self.device)
                    src_mask_val_artificial = torch.zeros_like(src_mask_val_complete, dtype=torch.bool)
                    src_mask_val_artificial.scatter_(1, missing_indices_val.unsqueeze(1), True)

                    # The "student" uses the artificial mask, the "teacher" uses the original (all False) mask.
                    output_val_incomplete = self.model.transformer.functional(params_star, True)(x=embs_val, src_mask=src_mask_val_artificial, return_pre_logits=True)
                    output_val_complete = self.model.transformer.functional(params_star, True)(x=embs_val, src_mask=src_mask_val_complete, return_pre_logits=True)
                    
                    loss_outer = self.loss(output_val_incomplete["logits"], y_val)
                    loss_dist_outer = F.mse_loss(output_val_complete["pre_logits"].detach(), output_val_incomplete["pre_logits"])
                    loss_meta_val = loss_outer + self.params_smil["alpha"] * loss_dist_outer
                    
                    loss = loss_meta_val
                    self.manual_backward(loss)
                    opt.step()
                    
                    self.log(f"train/loss_smil_meta_train", loss_meta_train.detach(), on_step=True, logger=True, sync_dist=True)
                    self.log(f"train/loss_smil_meta_val", loss_meta_val.detach(), on_step=True, logger=True, sync_dist=True)
                    self.log(f"train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

                    return { 
                        "loss": loss.detach(),
                        "logits": output_val_incomplete["logits"],
                        "y": y_val
                    }

        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        # additional arguments
        # BMML: Balanced Multimodal Learning
        if len(self.params_bmml) > 0:
            self.model.transformer.bmml_rebalancing_mode = self.bmml_rebalancing_mode

        if convert_logits == "mosi":
            y = y.squeeze()
        elif convert_logits == "ch_sims":
            five_classmapping = {
                -1.0: 0, -0.8: 0,
                -0.6: 1, -0.4: 1, -0.2: 1,
                0.0: 2,
                0.2: 3, 0.4: 3, 0.6: 3,
                0.8: 4, 1.0: 4
            }  # 0=negative, 1=weakly negative, 2=neutral, 3=weakly positive, 4=positive
            y = torch.tensor([five_classmapping.get(float(v), -1) for v in y], device=y.device, dtype=torch.long)
        else:
            # For single-label classification with CE loss, target needs to be 1D.
            # This check ensures we squeeze only for single-label cases.
            if self.params_mcr and not self.params_mcr["is_multilabel"]:
                if y.ndim > 1 and y.shape[1] == 1:
                    y = y.squeeze(1)

        output_dict = self.forward(x, y)
        logits = output_dict["logits"]

        loss = torch.tensor(0.0, device=logits.device)
        target = output_dict.get("target", y) # Use mixed target from model if available, else original y

        if len(self.params_avmc) > 0:            
            # Supervised Multitask Loss
            total_sup_loss = self.loss(logits, target)
            if "unimodal_logits" in output_dict:
                for i, u_logits in enumerate(output_dict["unimodal_logits"].items()):
                    total_sup_loss += self.loss(u_logits[1], target) * self.params_avmc["unimodal_loss_weights"][i]
            
            loss = total_sup_loss

            # Consistency Regularization Loss 
            consistency_weights = self.params_avmc["consistency_loss_weights"]
            consistency_loss_type = self.params_avmc["consistency_loss_type"]
            
            if consistency_loss_type == "mse":
                consistency_loss_fn = torch.nn.MSELoss()
            elif consistency_loss_type == "l1":
                consistency_loss_fn = torch.nn.L1Loss()
            elif consistency_loss_type == "kldiv":
                consistency_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
            
            total_consistency_loss = torch.tensor(0.0, device=logits.device)
            mixed_unimodal_logits = output_dict["unimodal_logits"]
            for mod_num, cons_target in output_dict["consistency_targets"].items():
                pred_logits = mixed_unimodal_logits[mod_num]
                
                detached_target = cons_target.detach()  # prevent gradients from flowing through the "teacher" path.
                
                if consistency_loss_type == 'kldiv':
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    consistency_loss = consistency_loss_fn(log_probs, detached_target)
                else:
                    consistency_loss = consistency_loss_fn(pred_logits, detached_target)
                
                total_consistency_loss += consistency_loss * consistency_weights[int(mod_num)]
            
            loss += total_consistency_loss
        else:
            # SimMLM: More vs. Fewer Modality (MoFe) ranking loss
            if "logits_plus" in output_dict and self.training:
                loss_task_plus = self.loss(output_dict["logits_plus"], target)
                loss_task_minus = self.loss(output_dict["logits_minus"], target)

                # MoFe ranking loss (Equation 3 from the paper)
                loss_mofe = torch.relu(loss_task_plus - loss_task_minus)
                
                mofe_lambda = self.params_simmlm.get("mofe_lambda", 0.1)

                # Total loss (Equation 4 from the paper)
                loss = loss_task_plus + loss_task_minus + mofe_lambda * loss_mofe

                # Log individual components for monitoring
                self.log(f"{set}/loss_task_plus", loss_task_plus.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f"{set}/loss_task_minus", loss_task_minus.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f"{set}/loss_mofe", loss_mofe.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            else:
                loss = self.loss(logits, target)

        # Apply distillation if teacher logits are provided by the model (e.g., ALBEF)
        if 'logits_m' in output_dict:
            logits_m = output_dict['logits_m']
            alpha = self.params_albef.get('alpha', 0.4)
            distill_temp = self.params_albef.get('distill_temp', 0.1)

            soft_target_downstream = F.softmax(logits_m / distill_temp, dim=1)
            distill_loss_downstream = -torch.sum(F.log_softmax(logits / distill_temp, dim=1) * soft_target_downstream, dim=1).mean()
            
            loss = (1 - alpha) * loss + alpha * distill_loss_downstream
        
        if self.params_gblend.get("mode") == "online" and "unimodal_logits" in output_dict:
            # Fused Loss
            if self.training:
                self.gblend_train_loss_fused.update(loss)
            else:
                self.gblend_val_loss_fused.update(loss)
            
            # Unimodal Losses
            for i, u_logits in enumerate(output_dict["unimodal_logits"]):
                uni_loss = self.loss(u_logits.squeeze(), y)
                if self.training:
                    if self.gblend_train_losses_uni: self.gblend_train_losses_uni[i].update(uni_loss)
                else:
                    if self.gblend_val_losses_uni: self.gblend_val_losses_uni[i].update(uni_loss)

        self.log(f"{set}/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        additional_loss_sum = torch.tensor(0.0, device=logits.device)

        # Method-sepcific auxiliary losses
        if "losses" in output_dict:
            for loss_name, loss_value in output_dict["losses"].items():
                # Interleaving logic for EBR, as per Appendix C.1
                if self.training and loss_name == "ebr":
                    # E.g., for interleave_epochs=10, Epochs 0-9: L_sem only. Epochs 10-19: L_sem + L_md.
                    if self.params_ebr["interleave_epochs"] > 0:
                        if (self.current_epoch % (2 * self.params_ebr["interleave_epochs"])) < self.params_ebr["interleave_epochs"]:
                            self.log(f"{set}/{loss_name}", loss_value.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                            continue 

                self.log(f"{set}/{loss_name}", loss_value, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                additional_loss_sum += loss_value

        if "unimodal_logits" in output_dict:

            # ARL: Asymmetric Representation Learning
            if 'arl_coeffs' in output_dict:
                method_loss = 0.0
                for u_logits in output_dict["unimodal_logits"]:
                    method_loss += self.loss(u_logits, y)
                loss_weight = self.params_arl["unimodal_loss_weight"]

                self.log(f"{set}/loss_arl", method_loss.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                additional_loss_sum += method_loss * loss_weight

            # PDF: Predictive Dynamic Fusion
            if "pdf" in output_dict:
                unimodal_logits = output_dict["unimodal_logits"]
                p_trues_pred = output_dict["p_trues"]
                num_modalities = len(unimodal_logits)
                
                total_p_true_loss = torch.tensor(0.0, device=y.device)
                total_unimodal_loss = torch.tensor(0.0, device=y.device)

                is_bce = isinstance(self.loss, (nn.BCEWithLogitsLoss, nn.BCELoss, WeightedNaNBCEWithLogitsLoss))

                if self.params_pdf["p_true_loss_fn"] == "mse":
                    loss_fn_ptrue = F.mse_loss
                else: # default to l1
                    loss_fn_ptrue = F.l1_loss 
                
                for i in range(num_modalities):
                    u_logits = unimodal_logits[i]
                    
                    total_unimodal_loss += self.loss(u_logits, y)

                    if is_bce:
                        # For BCE/multilabel, p_true is the average of p for positive labels and (1-p) for negative labels
                        u_probs = torch.sigmoid(u_logits)
                        p_true_target = torch.where(y == 1, u_probs, 1 - u_probs)
                        
                        # Handle NaNs in targets by ignoring them in the mean
                        nan_mask = torch.isnan(y)
                        if nan_mask.any():
                            p_true_target[nan_mask] = float('nan')
                            p_true_target = torch.nanmean(p_true_target, dim=-1, keepdim=True)
                        else:
                            p_true_target = torch.mean(p_true_target, dim=-1, keepdim=True)
                    else: # CE
                        u_probs = F.softmax(u_logits, dim=-1)
                        y_target_ce = y.squeeze() if y.ndim > 1 else y
                        y_target_ce = y_target_ce.unsqueeze(1) if y_target_ce.ndim == 1 else y_target_ce
                        p_true_target = u_probs.gather(1, y_target_ce)
                    
                    # Detach target to prevent gradients from flowing back through the classifier into the p_head loss
                    p_true_target_detached = p_true_target.detach()

                    valid_target_mask = ~torch.isnan(p_true_target_detached)
                    p_true_loss = loss_fn_ptrue(
                        p_trues_pred[i][valid_target_mask],
                        p_true_target_detached[valid_target_mask]
                    )
                    if torch.isnan(p_true_loss):
                        p_true_loss = torch.tensor(0.0, device=y.device)

                    total_p_true_loss += p_true_loss

                avg_p_true_loss = total_p_true_loss / num_modalities

                self.log(f"{set}/loss_pdf_ptrue", avg_p_true_loss.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f"{set}/loss_pdf_unimodal", total_unimodal_loss.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                additional_loss_sum += avg_p_true_loss * self.params_pdf["loss_weight"]
                additional_loss_sum += total_unimodal_loss * self.params_pdf.get("unimodal_loss_weight", 1.0)

        total_loss = loss + additional_loss_sum
        
        if self.manual_opt and self.training:
            opt = self.optimizers()
            # DGL: Disentangled Gradient Learning
            if 'dgl' in output_dict:
                unimodal_loss = 0.0
                for u_logits in output_dict["unimodal_logits"]:
                    unimodal_loss += self.loss(u_logits, y)
                self.log(f"{set}/loss_dgl", unimodal_loss, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                unimodal_loss *= self.params_dgl["unimodal_loss_weight"]
                total_loss += unimodal_loss

                # Backward pass for unimodal loss to update encoders ONLY
                self.manual_backward(unimodal_loss, retain_graph=True)

                # Manually set fusion module gradients to None to prevent updates from unimodal loss.
                for param in self.model.transformer.transformer_cls_pipeline.parameters():
                    param.grad = None

                # Backward pass for multimodal loss to update fusion module ONLY.
                self.manual_backward(loss)

            # OGM: On-the-fly Gradient Modulation
            # also see https://github.com/GeWu-Lab/OGM-GE_CVPR2022/blob/main/main.py#L132 for official implementation
            elif "ogm" in output_dict:
                self.manual_backward(total_loss)
                num_modalities = len(self.model.encoders.encoders)

                coeffs = output_dict["ogm"]["modulation_coeffs"]
                if len(coeffs) == num_modalities:
                    for i in range(num_modalities):
                        k_mod = coeffs[i]
                        encoder_module = self.model.encoders.encoders[i]
                        # Scale gradients for parameters of this specific encoder
                        for param in encoder_module.parameters():
                            if param.grad is not None:
                                param.grad.mul_(k_mod)
                        if self.params_ogm["use_ge"]:
                            for param in encoder_module.parameters(): 
                                if param.grad is not None:
                                    # Calculate std dev of the current param's gradient
                                    grad_std = param.grad.std().item()
                                    # Add small epsilon for numerical stability if std dev is zero
                                    noise_std = grad_std + 1e-8
                                    # Sample noise using this std dev and add
                                    noise = torch.randn_like(param.grad) * noise_std
                                    param.grad.add_(noise)

            # G-Blend: Gradient Blending
            elif "gblend" in output_dict:
                # Weights are pre-calculated in on_fit_start
                weights = self.gblend_weights.to(self.device)
                
                # The last weight is for the fused head.
                gblend_task_loss = weights[-1] * loss

                # The first N weights are for the unimodal heads
                for i, u_logits in enumerate(output_dict["unimodal_logits"]):
                    gblend_task_loss += weights[i] * self.loss(u_logits.squeeze(), y)
                
                final_loss = gblend_task_loss + additional_loss_sum
                self.manual_backward(final_loss)
                total_loss = final_loss

            # ARL: Asymmetric Representation Learning
            elif "arl_coeffs" in output_dict:
                self.manual_backward(total_loss)
                num_modalities = len(self.model.encoders.encoders)

                coeffs = output_dict["arl_coeffs"]
                if len(coeffs) == num_modalities:
                    for i in range(num_modalities):
                        k_mod = coeffs[i]
                        encoder_module = self.model.encoders.encoders[i]
                        for param in encoder_module.parameters():
                            if param.grad is not None:
                                # Incorporate original gradient as a residual component
                                # new_grad = grad + coeff * grad = grad * (1 + coeff)
                                param.grad.mul_(1 + k_mod)

            # MCR: Multimodal Competition Regularizer
            elif len(self.params_mcr) > 1:
                self.manual_backward(total_loss, retain_graph=True)

                unfused_reps = output_dict["unimodal_reps"]
                num_modalities = len(unfused_reps)
                bs = unfused_reps[0].size(0)

                # Select activation based on task type (single-label vs multi-label)
                if self.params_mcr["is_multilabel"]:
                    prob_fn = torch.sigmoid
                    log_prob_fn = F.logsigmoid
                else:
                    prob_fn = lambda x: F.softmax(x, dim=-1)
                    log_prob_fn = lambda x: F.log_softmax(x, dim=-1)

                # L_MIPD computation
                log_p_orig = log_prob_fn(logits)
                mipd_losses = []
                
                for _ in range(self.params_mcr["num_permutations"]):
                    current_iter_losses = []
                    for i in range(num_modalities):
                        permuted_reps = []
                        for j in range(num_modalities):
                            if i == j:
                                perm_indices = torch.randperm(bs).to(unfused_reps[j].device)
                                permuted_reps.append(unfused_reps[j][perm_indices])
                            else:
                                permuted_reps.append(unfused_reps[j])
                        
                        permuted_input = torch.cat(permuted_reps, dim=1)
                        permuted_logits = self.model.transformer.fusion(permuted_input)
                        log_p_perm = log_prob_fn(permuted_logits)

                        # JSD approximation with KL divergence
                        p_orig = prob_fn(logits)
                        p_perm = prob_fn(permuted_logits)
                        
                        m = 0.5 * (p_orig + p_perm)
                        log_m = torch.log(m + 1e-8) # Add epsilon for stability

                        jsd = 0.5 * (F.kl_div(log_m, p_orig, reduction='batchmean', log_target=False) + \
                                     F.kl_div(log_m, p_perm, reduction='batchmean', log_target=False))
                        
                        current_iter_losses.append(-jsd)

                    if not mipd_losses:
                        mipd_losses = current_iter_losses
                    else:
                        for i in range(num_modalities):
                            mipd_losses[i] += current_iter_losses[i]

                # Average the MIPD losses over the number of permutations
                mipd_losses = [loss / self.params_mcr["num_permutations"] for loss in mipd_losses]

                for i in range(num_modalities):
                    self.log(f"{set}/loss_mcr_mipd_{i}", mipd_losses[i].detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

                # L_Con, L_CEB, and L_uni
                avg_unfused_reps = [torch.mean(mod, dim=1) for mod in unfused_reps]
                loss_con = self.l_con(avg_unfused_reps, y)
                loss_ceb = self.l_ceb(avg_unfused_reps, y) 

                # Unimodal Task Loss Calculation
                total_unimodal_loss = 0.0
                unimodal_weights = self.params_mcr["loss_weights"]["uni"]
                unimodal_logits = output_dict["unimodal_logits"]
                assert len(unimodal_weights) == num_modalities, "Number of unimodal loss weights must match number of modalities."

                for i in range(num_modalities):
                    uni_loss = self.loss(unimodal_logits[i], y)
                    self.log(f"{set}/loss_mcr_uni_{i}", uni_loss.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                    total_unimodal_loss += uni_loss * unimodal_weights[i]

                self.log(f"{set}/loss_mcr_con", loss_con.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
                self.log(f"{set}/loss_mcr_ceb", loss_ceb.detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

                # Add L_Con, L_CEB, and L_uni to gradients
                loss_weights = self.params_mcr["loss_weights"]
                mcr_aux_loss = (loss_weights["con"] * loss_con + loss_weights["ceb"] * loss_ceb) + total_unimodal_loss
                self.manual_backward(mcr_aux_loss, retain_graph=True)

                # Game-theoretic update strategy
                strategy = self.params_mcr["strategy"]

                # collaborative strategy
                if strategy == "Collaborative":
                    collaborative_loss = sum(mipd_losses)
                    self.manual_backward(collaborative_loss * loss_weights["mipd"])
                # independent and greedy strategy
                else:
                    for i in range(num_modalities):
                        encoder_module = self.model.encoders.encoders[i]
                        
                        if strategy == "Greedy":
                            # Each encoder minimizes its own L_MIPD (maximizes importance)
                            # while maximizing all others' L_MIPD (minimizes their importance).
                            game_loss = (mipd_losses[i] - sum(mipd_losses[j] for j in range(num_modalities) if i != j))
                        elif strategy == "Independent":
                            # Each encoder only minimizes its own L_MIPD.
                            game_loss = mipd_losses[i]
                        else:
                            raise ValueError(f"Unknown MCR strategy: {strategy}")

                        # Determine if we need to retain the graph for the next loop iteration
                        should_retain_graph = i < num_modalities - 1

                        # Filter for only the parameters that require gradients (e.g., for frozen encoders)
                        trainable_params = [p for p in encoder_module.parameters() if p.requires_grad]
                        if not trainable_params:
                            continue

                        # Calculate gradients for this encoder's game loss on its trainable parameters.
                        game_grads = torch.autograd.grad(
                            game_loss * loss_weights["mipd"], 
                            trainable_params,
                            retain_graph=should_retain_graph,
                            allow_unused=True # In case some params are not used in game loss path
                        )

                        # Manually add the game gradients to the existing gradients
                        for param, grad in zip(trainable_params, game_grads):
                            if grad is not None and param.grad is not None:
                                param.grad.add_(grad)

            # BMML: Balanced Multi-modal Learning
            elif "bmml" in output_dict:
                num_modalities = len(self.model.encoders.encoders)
                unimodal_weight = self.params_bmml["unimodal_loss_weight"]

                bmml_total_loss = loss
                for u_logits in output_dict["unimodal_logits"]:
                    bmml_total_loss += self.loss(u_logits, y) * unimodal_weight
                
                self.manual_backward(bmml_total_loss)

                if self.trainer.current_epoch >= self.params_bmml["warmup_epochs"]:
                    
                    def get_params_and_grads(module):
                        params = [p for p in module.parameters() if p.grad is not None and p.requires_grad]
                        grads = [p.grad for p in params]
                        return params, grads

                    def compute_mu(params, grads):
                        param_norm_sq = sum([p.norm().pow(2) for p in params])
                        grad_norm_sq = sum([g.norm().pow(2) for g in grads])
                        return grad_norm_sq / (param_norm_sq + 1e-8)

                    # Effective update for the shared fusion transformer ('bypass' branch, θ')
                    transformer_params, transformer_grads = get_params_and_grads(self.model.transformer)
                    mu_prime = compute_mu(transformer_params, transformer_grads)

                    # Effective update for each unimodal encoder ('main' branch, θ)
                    mu_thetas = []
                    for i in range(num_modalities):
                        encoder_params, encoder_grads = get_params_and_grads(self.model.encoders.encoders[i])
                        mu_thetas.append(compute_mu(encoder_params, encoder_grads))
                    
                    for i in range(num_modalities):
                        self.bmml_M_accumulators["theta"][i] += mu_thetas[i].item()
                        # The 'prime' accumulator represents the shared fusion part for all modalities
                        self.bmml_M_accumulators["theta_prime"][i] += mu_prime.item()

                    # Calculate conditional learning speeds S_i for all modalities
                    cond_speeds = []
                    for i in range(num_modalities):
                        s_i = torch.log(torch.tensor(self.bmml_M_accumulators["theta_prime"][i])) - torch.log(torch.tensor(self.bmml_M_accumulators["theta"][i]))
                        cond_speeds.append(s_i)
                        self.log(f"{set}/s_m{i}", s_i.item(), on_step=True, logger=True, sync_dist=True)
                    
                    # dspeed is the difference between the fastest and slowest learning modalities
                    s_min = min(cond_speeds)
                    s_max = max(cond_speeds)
                    dspeed = s_max - s_min
                    modality_to_boost_idx = cond_speeds.index(s_min)
                    
                    self.log(f"{set}/dspeed", dspeed.item(), on_step=True, logger=True, sync_dist=True)

                    if self.bmml_rebalancing_counter > 0:
                        self.bmml_rebalancing_counter -= 1
                        if self.bmml_rebalancing_counter == 0:
                            self.bmml_rebalancing_mode = 'none'
                    
                    elif abs(dspeed.item()) > self.params_bmml["alpha"]:
                        self.bmml_rebalancing_counter = self.params_bmml["q"]
                        # Boost the modality with the minimum conditional learning speed
                        self.bmml_rebalancing_mode = f'm{modality_to_boost_idx}'

            # MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance
            elif "mmpareto" in output_dict:
                """ 
                https://arxiv.org/abs/2405.17730
                https://github.com/GeWu-Lab/MMPareto_ICML2024 
                """
                opt.zero_grad()
                multimodal_loss = loss
                unimodal_losses = [self.loss(u_logits, y) for u_logits in output_dict["unimodal_logits"]]

                all_params = list(self.parameters())
                encoder_param_sets = [set(enc.parameters()) for enc in self.model.encoders.encoders]
                
                # Calculate and assign gradients for non-encoder parameters (e.g., fusion module) from the multimodal loss
                non_encoder_params = [p for p in all_params if not any(p in s for s in encoder_param_sets) and p.requires_grad]
                if non_encoder_params:
                    non_encoder_grads = torch.autograd.grad(multimodal_loss, non_encoder_params, retain_graph=True, allow_unused=True)
                    for param, grad in zip(non_encoder_params, non_encoder_grads):
                        if grad is not None:
                            param.grad = grad.clone()

                # Handle each encoder with MMPareto logic
                for i, encoder in enumerate(self.model.encoders.encoders):
                    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
                    if not encoder_params:
                        continue
                    
                    # Get multimodal and unimodal gradients for this encoder
                    g_m_tensors = torch.autograd.grad(multimodal_loss, encoder_params, retain_graph=True, allow_unused=True)
                    g_u_tensors = torch.autograd.grad(unimodal_losses[i], encoder_params, retain_graph=True, allow_unused=True)

                    # Replace None grads with zeros
                    g_m_tensors = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_m_tensors, encoder_params)]
                    g_u_tensors = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_u_tensors, encoder_params)]
                    
                    g_m_flat = torch.cat([g.reshape(-1) for g in g_m_tensors])
                    g_u_flat = torch.cat([g.reshape(-1) for g in g_u_tensors])
                    
                    cos_sim = F.cosine_similarity(g_m_flat, g_u_flat, dim=0)
                    
                    if cos_sim >= 0:
                        # Non-conflict case: Use equal weights, which corresponds to a simple sum
                        alphas = np.array([0.5, 0.5])
                    else:
                        # Conflict case: Solve for Pareto-optimal weights
                        grads_for_solver = [
                            [g.detach() for g in g_m_tensors],
                            [g.detach() for g in g_u_tensors]
                        ]
                        alphas, _ = MinNormSolver.find_min_norm_element(grads_for_solver)

                    # Calculate the new gradient direction based on the weights
                    # The factor of 2 keeps the scale consistent with a simple sum when alphas are 0.5
                    h_prime_flat = 2 * alphas[0] * g_m_flat + 2 * alphas[1] * g_u_flat

                    # Calculate the target magnitude from the uniform baseline sum (g_m + g_u)
                    uniform_sum_flat = g_m_flat + g_u_flat
                    target_magnitude = torch.norm(uniform_sum_flat)

                    # Rescale to match target magnitude and apply gamma, as per Algorithm 1
                    h_prime_norm = torch.norm(h_prime_flat)
                    rescale_factor = target_magnitude / (h_prime_norm + 1e-8)
                    
                    # Note: The paper suggests gamma > 1 for magnitude enhancement (e.g., 1.5).
                    gamma = self.params_mmpareto.get("gamma", 1.5)
                    final_grad_flat = h_prime_flat * rescale_factor * gamma

                    # Apply the final calculated gradient to the encoder parameters
                    offset = 0
                    for param in encoder_params:
                        numel = param.numel()
                        if numel > 0:
                            param.grad = final_grad_flat[offset:offset+numel].view_as(param).clone()
                        offset += numel
            
            opt.step()
            return {  
                "loss": total_loss.detach(),
                "logits": logits.detach(),
                "y": y,
            }  # detached returns when already backpropped
        
        return {
            "loss": total_loss, 
            "logits": logits,
            "y": y,
        }

    def on_train_start(self):
        self.optimizers().train() if "schedulefree" in self.params_optimizer["name"] else None
        if len(self.params_smil) > 0 and not self.smil_meta_iters_setup:
            print("SMIL: Setting up meta-learning iterators and wrapping model.")
            # It's recommended to install MetaNN: pip install MetaNN==0.2.5
            if not isinstance(self.model.transformer, ProtoModule):
                self.model.transformer = ProtoModule(self.model.transformer)
            
            self.meta_train_loader_iter = iter(self.trainer.datamodule.train_dataloader())
            self.meta_val_loader_iter = iter(self.trainer.datamodule.meta_val_dataloader())
            self.smil_meta_iters_setup = True
        
    def on_validation_start(self):
        self.optimizers().eval() if "schedulefree" in self.params_optimizer["name"] else None
        # For EBR, compute the substitution ranking before the first validation run.
        if isinstance(self.model.transformer, Explicit_Basis_Reallocation_Transformer) and self.model.transformer.modality_ranking is None:
            self._compute_and_set_ebr_ranking(self.trainer.datamodule.val_dataloader())
        
    def on_validation_end(self):
        self.optimizers().train() if "schedulefree" in self.params_optimizer["name"] else None
    
    def training_epoch_end(self, outputs):
        if self.current_epoch == 0 and self.epoch_start_time is not None and self.trainer.is_global_zero:
            epoch_duration = time.time() - self.epoch_start_time
            if self.logger:
                self.logger.log_metrics({"train/epoch_duration_seconds": epoch_duration}, step=self.global_step)
            self.epoch_start_time = None

        if self.params_gblend.get("mode") == "online":
            num_modalities = self.params_gblend["num_modalities"]
            train_losses = []

            # Unimodal heads
            if self.gblend_train_losses_uni:
                for i in range(num_modalities):
                    loss_val = self.gblend_train_losses_uni[i].compute()
                    self.log(f'train/gblend_loss_uni_{i}_epoch', loss_val, sync_dist=True)
                    train_losses.append(loss_val.item())
                    self.gblend_train_losses_uni[i].reset()

            # Fused head
            if self.gblend_train_loss_fused:
                loss_val = self.gblend_train_loss_fused.compute()
                self.log('train/gblend_loss_fused_epoch', loss_val, sync_dist=True)
                train_losses.append(loss_val.item())
                self.gblend_train_loss_fused.reset()
            
            self.current_train_gblend_losses = train_losses

    def validation_epoch_end(self, outputs):
        if self.params_gblend.get("mode") != "online":
            return

        if self.params_gblend:
            num_modalities = self.params_gblend["num_modalities"]
            num_heads = num_modalities + 1
            
            val_gblend_losses = []
            if self.gblend_val_losses_uni:
                for i in range(num_modalities):
                    loss_val = self.gblend_val_losses_uni[i].compute()
                    self.log(f'val/gblend_loss_uni_{i}_epoch', loss_val, sync_dist=True)
                    val_gblend_losses.append(loss_val.item())
                    self.gblend_val_losses_uni[i].reset()
            if self.gblend_val_loss_fused:
                loss_val = self.gblend_val_loss_fused.compute()
                self.log('val/gblend_loss_fused_epoch', loss_val, sync_dist=True)
                val_gblend_losses.append(loss_val.item())
                self.gblend_val_loss_fused.reset()
                
            update_freq = self.params_gblend["update_freq"]
            if (self.current_epoch + 1) % update_freq != 0 or self.trainer.sanity_checking:
                return

            if not self.current_train_gblend_losses or not val_gblend_losses:
                return  # prevents errors from validation-only runs (e.g., sanity check).

            current_epoch_losses = {
                'train_gblend': self.current_train_gblend_losses,
                'val_gblend': val_gblend_losses
            }
            
            self.gblend_history.append(current_epoch_losses)

            if len(self.gblend_history) < 2:
                return

            prev_losses = self.gblend_history[-2]
            current_losses = self.gblend_history[-1]

            raw_weights = []
            for i in range(num_heads):
                delta_g = prev_losses['val_gblend'][i] - current_losses['val_gblend'][i]
                
                delta_o_train = current_losses['train_gblend'][i] - prev_losses['train_gblend'][i]
                delta_o_val = current_losses['val_gblend'][i] - prev_losses['val_gblend'][i]
                delta_o = delta_o_train - delta_o_val

                weight = max(delta_g, 1e-8) / (delta_o**2 + 1e-8)
                raw_weights.append(weight)

            new_weights = torch.tensor(raw_weights, dtype=torch.float32)
            new_weights = new_weights / torch.sum(new_weights)
            self.gblend_weights = new_weights
            
            for i in range(num_modalities):
                self.log(f"gblend/weight_uni_{i}", self.gblend_weights[i], rank_zero_only=True)
            self.log(f"gblend/weight_fused", self.gblend_weights[-1], rank_zero_only=True)

    def on_test_start(self):
        self.optimizers().eval() if "schedulefree" in self.params_optimizer["name"] else None
        # EBR: ensure the substitution ranking is computed before the test run.
        if isinstance(self.model.transformer, Explicit_Basis_Reallocation_Transformer) and self.model.transformer.modality_ranking is None:
            print("EBR: Computing modality ranking on validation set before testing.")
            self._compute_and_set_ebr_ranking(self.trainer.datamodule.val_dataloader())

    def configure_optimizers(self):
        if self.params_optimizer["name"] == "schedulefree_adamw":
            optimizer = schedulefree.AdamWScheduleFree(
                self.parameters(), 
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"], 
                eps=self.params_optimizer["eps"],
                warmup_steps=self.params_optimizer["warmup_steps"],
                betas=self.params_optimizer["betas"]
            )
        elif self.params_optimizer["name"] == "schedulefree_sgd":
            optimizer = schedulefree.SGDScheduleFree(
                self.parameters(), 
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"], 
                momentum=self.params_optimizer["momentum"],
                warmup_steps=self.params_optimizer["warmup_steps"],
            )
        elif self.params_optimizer["name"] == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.params_optimizer["lr"],
                weight_decay=self.params_optimizer["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer {self.params_optimizer['name']} not supported.")
        
        return optimizer



class NaNMultilabelAUROC(Metric):
    
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        num_labels: int,
        average: str = "macro",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.average = average
        
        self.ignore_index = num_labels + 1
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

        #self.preds = []
        #self.target = []

    def update(self, preds, target) -> None:
        mask = torch.isnan(target)
        target = target.clone()

        target[mask] = self.ignore_index

        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        target = target.to(torch.long)

        return auroc(preds, target, task="multilabel", num_classes=self.num_labels, num_labels=self.num_labels, average=self.average, ignore_index=self.ignore_index)


def get_input(name_dataset, dataloader_output):
    if name_dataset == "mimic_haim":
        x = [dataloader_output["image"], dataloader_output["lab"]]
    elif name_dataset == "mimic_symile":
        x = [
            dataloader_output["image"],
            dataloader_output["lab"], 
            dataloader_output["ecg"]
        ]
    elif name_dataset == "vgg_sound":
        x = [dataloader_output["vision"], dataloader_output["audio"]]
    elif name_dataset == "crema_d":
        x = [dataloader_output["vision"], dataloader_output["audio"]]
    elif name_dataset == "fmnist" or name_dataset == "mnist":
        if len(dataloader_output[0]) == 2:
            x = [dataloader_output[0][0].to(torch.float32), 
                 dataloader_output[0][1].to(torch.float32)]
        if len(dataloader_output[0]) == 4:
            x = [dataloader_output[0][0].to(torch.float32),
                 dataloader_output[0][1].to(torch.float32),
                 dataloader_output[0][2].to(torch.float32),
                 dataloader_output[0][3].to(torch.float32)]
        x_missing = None 
    elif name_dataset == "mosi" or name_dataset == "mosei": 
        x = [dataloader_output["language"],
             dataloader_output["vision"],
             dataloader_output["audio"]]
        return x
    elif name_dataset == "ch_sims" or name_dataset == "ch_sims_v2":
        x = [dataloader_output["video"],
             dataloader_output["audio"],
             dataloader_output["text"]]
    elif name_dataset == "vision_touch":
        x = [dataloader_output["image"],
             dataloader_output["proprio"],
             dataloader_output["force"]]
    elif name_dataset == "kinetics_400" or name_dataset == "kinetics_600" or name_dataset == "kinetics_700":
        x = [dataloader_output["vision"],
             dataloader_output["audio"]]
    elif name_dataset == "inspect":
        x = [dataloader_output["image"],
             dataloader_output["ehr"]]
    else: 
        raise NotImplementedError

    return x

def get_target(name_dataset, dataloader_output):
    if name_dataset == "mimic_haim" or name_dataset == "mimic_symile":
        y = dataloader_output["target"]
    elif name_dataset == "fmnist" or name_dataset == "mnist":
        y = dataloader_output[1]
    elif name_dataset == "mosi" or name_dataset == "mosei":
        y = dataloader_output["label"]
    elif name_dataset == "vgg_sound":
        y = dataloader_output["target"]
    elif name_dataset == "crema_d":
        y = dataloader_output["target"]
    elif name_dataset == "ch_sims" or name_dataset == "ch_sims_v2":
        y = dataloader_output["label"]
    elif name_dataset == "vision_touch":
        y = dataloader_output["contact_next"]
    elif name_dataset == "kinetics_400" or name_dataset == "kinetics_600" or name_dataset == "kinetics_700":
        y = dataloader_output["target"]
    elif name_dataset == "inspect":
        y = dataloader_output["target"]
    else:
        raise NotImplementedError
    return y