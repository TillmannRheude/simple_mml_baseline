import torch 
import schedulefree
import warnings
import time

import torch.nn as nn 
import torch.nn.functional as F
import pytorch_lightning as pl 
import numpy as np

from unittest.mock import patch
from lightning.fabric.utilities.throughput import measure_flops
from torchmetrics import Metric, MeanMetric
from torchmetrics.functional import auroc
from torchmetrics.utilities import dim_zero_cat

from codefiles.methods.aug.aug import AUG_Transformer
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss
from codefiles.methods.mmpareto.min_norm_solvers import MinNormSolver

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
        params_omib: dict = {
            "warmup_epochs": 1,
        },
        params_aug: dict = {
            "check_interval": 1,
            "threshold": 0.1,
            "confidence_coeff": 1.0,
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
        self.params_mmpareto = params_mmpareto
        self.params_bmml = params_bmml
        self.params_gblend = params_gblend
        self.params_pdf = params_pdf
        self.params_omib = params_omib
        self.params_aug = params_aug

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
                    if self.dataset in ["ch_sims", "ch_sims_v2"]:
                        five_classmapping = { -1.0: 0, -0.8: 0, -0.6: 1, -0.4: 1, -0.2: 1, -0.1: 1, 0.0: 2, 0.2: 3, 0.4: 3, 0.6: 3, 0.8: 4, 1.0: 4 }
                        y = torch.tensor([five_classmapping.get(float(v), -1) for v in y], device=y.device, dtype=torch.long)

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
                        if self.dataset in ["ch_sims", "ch_sims_v2"]:
                            five_classmapping = { -1.0: 0, -0.8: 0, -0.6: 1, -0.4: 1, -0.2: 1, -0.1: 1, 0.0: 2, 0.1: 2, 0.2: 3, 0.4: 3, 0.6: 3, 0.8: 4, 1.0: 4 }
                            y = torch.tensor([five_classmapping.get(float(v), -1) for v in y], device=y.device, dtype=torch.long)
                        
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

    def _log_flops_once(self):
        """Calculates and logs the model's FLOPs for a single forward pass."""
        try:
            dataloader = self.trainer.datamodule.train_dataloader()
            batch = next(iter(dataloader))
            batch = self.transfer_batch_to_device(batch, self.device, 0)
            x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)
            batch_size = x[0].shape[0] if isinstance(x, list) and len(x) > 0 else x.shape[0]

            forward_fn = lambda: self.model(x, y)
            
            # For models with internal no_grad blocks, FLOPs calculation can fail.
            # We temporarily patch torch.no_grad to torch.enable_grad to allow the tracer
            # to see all operations. 
            with patch('torch.no_grad', torch.enable_grad):
                total_flops_for_batch = measure_flops(self.model, forward_fn)

            gflops_per_instance = (total_flops_for_batch / batch_size) / 1e9
            
            if self.logger:
                self.logger.log_metrics({"train/GFlops": gflops_per_instance}, step=0)
            print(f"Model GFlops (per instance): {gflops_per_instance:.2f}")

        except Exception as e:
            warnings.warn(f"Could not calculate Flops: {e}")

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
        
        x, y = get_input(self.dataset, batch), get_target(self.dataset, batch)

        # BMML: Balanced Multimodal Learning
        if len(self.params_bmml) > 0:
            self.model.transformer.bmml_rebalancing_mode = self.bmml_rebalancing_mode

        # Convert logits for MOSI / CHSims
        if convert_logits == "mosi":
            y = y.squeeze()
        elif convert_logits == "ch_sims":
            five_classmapping = {
                -1.0: 0, -0.8: 0,
                -0.6: 1, -0.4: 1, -0.2: 1, -0.1: 1,
                0.0: 2,
                0.2: 3, 0.4: 3, 0.6: 3,
                0.8: 4, 1.0: 4
            }  # 0=negative, 1=weakly negative, 2=neutral, 3=weakly positive, 4=positive
            y_mapped = []
            for v in y:
                fv = float(v.detach().cpu())
                fv = round(fv, 1)
                y_mapped.append(five_classmapping.get(fv, -1))
            y = torch.tensor(y_mapped, device=y.device, dtype=torch.long)

        # Forward pass
        output_dict = self.forward(x, y)
        logits = output_dict["logits"]

        # Calculate Loss 
        loss = torch.tensor(0.0, device=logits.device)
        target = output_dict.get("target", y) # Use updated target from model if available, else original y
        if "aug" in output_dict:
            loss = output_dict["loss"]
        else:
            loss = self.loss(logits, target)
        self.log(f"{set}/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # G-Blend 
        if self.params_gblend.get("mode") == "online" and "unimodal_logits" in output_dict:
            # Fused Loss
            if self.training:
                self.gblend_train_loss_fused.update(loss)
            else:
                self.gblend_val_loss_fused.update(loss)
            
            # Unimodal Losses
            for i, u_logits in enumerate(output_dict["unimodal_logits"]):
                uni_loss = self.loss(u_logits.squeeze(1), target)
                if self.training:
                    if self.gblend_train_losses_uni: self.gblend_train_losses_uni[i].update(uni_loss)
                else:
                    if self.gblend_val_losses_uni: self.gblend_val_losses_uni[i].update(uni_loss)

        # Calculate method-specific additional losses
        additional_loss_sum = torch.tensor(0.0, device=logits.device)
        if "losses" in output_dict:
            for loss_name, loss_value in output_dict["losses"].items():
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
                    gblend_task_loss += weights[i] * self.loss(u_logits.squeeze(1), y)
                
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
            
            # AUG
            elif "aug" in output_dict:
                opt.zero_grad()
                total_loss = output_dict["loss"]
                total_loss.backward()

            opt.step()

            return {  
                "loss": total_loss.detach(),
                "logits": logits.detach(),
                "y": y,
            }  
        
        return {
            "loss": total_loss, 
            "logits": logits,
            "y": y,
        }

    def on_train_start(self):
        self.optimizers().train() if "schedulefree" in self.params_optimizer["name"] else None
        
    def on_train_epoch_start(self):
        self.optimizers().train() if "schedulefree" in self.params_optimizer["name"] else None
        
    def on_validation_start(self):
        self.optimizers().eval() if "schedulefree" in self.params_optimizer["name"] else None
        
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

        if isinstance(self.model.transformer, AUG_Transformer):
            if self.current_epoch % self.params_aug["check_interval"] == 0:
                performance_ratios = self.model.transformer.calculate_performance_ratios()
                weakest_modality_idx = performance_ratios.index(min(performance_ratios))
                strongest_modality_idx = performance_ratios.index(max(performance_ratios))

                gap = performance_ratios[strongest_modality_idx] - (self.params_aug["confidence_coeff"] * performance_ratios[weakest_modality_idx])
                if gap > self.params_aug["threshold"]:
                    self.model.transformer.add_layer(weakest_modality_idx)
                    print(f"Added layer to modality {weakest_modality_idx}.")
                else:
                    print(f"No layer added due to small difference in performance ratios. Difference: {performance_ratios[strongest_modality_idx] - performance_ratios[weakest_modality_idx]}")

            self.model.transformer.reset_scores_and_t_stats()

    def validation_epoch_end(self, outputs):
        if self.params_gblend:
            if self.params_gblend["mode"] != "online":
                return
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
        if "schedulefree" in self.params_optimizer["name"]:
            opts = self.optimizers()
            if isinstance(opts, list):
                for opt in opts:
                    opt.eval()
            else:
                opts.eval()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        if isinstance(self.model.transformer, AUG_Transformer):
            max_layers = {} # mod_idx -> max_layer_idx
            
            for key in state_dict.keys():
                if "additional_layers_modality" in key:
                    parts = key.split(".")
                    # finding index of 'additional_layers_modality'
                    try:
                        idx = parts.index("additional_layers_modality")
                        if idx + 2 < len(parts):
                            mod_idx = int(parts[idx+1])
                            layer_idx = int(parts[idx+2])
                            
                            if mod_idx not in max_layers:
                                max_layers[mod_idx] = -1
                            max_layers[mod_idx] = max(max_layers[mod_idx], layer_idx)
                    except (ValueError, IndexError):
                        continue
            
            # Add layers
            for mod_idx, max_layer_idx in max_layers.items():
                current_count = len(self.model.transformer.additional_layers_modality[mod_idx])
                target_count = max_layer_idx + 1
                
                params_added = 0
                while current_count < target_count:
                    self.model.transformer.add_layer(mod_idx)
                    current_count += 1
                    params_added += 1
                
                if params_added > 0:
                    print(f"Restored {params_added} dynamic layers for modality {mod_idx} from checkpoint.")

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
                # momentum=self.params_optimizer["momentum"],
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



""" Helpers """ 
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
    elif name_dataset == "crema_d":
        x = [dataloader_output["vision"], dataloader_output["audio"]]
    elif name_dataset == "mosi" or name_dataset == "mosei": 
        x = [dataloader_output["language"],
             dataloader_output["vision"],
             dataloader_output["audio"]]
        return x
    elif name_dataset == "ch_sims" or name_dataset == "ch_sims_v2":
        x = [dataloader_output["video"],
             dataloader_output["audio"],
             dataloader_output["text"]]
    elif name_dataset == "inspect":
        x = [dataloader_output["image"],
             dataloader_output["ehr"]]
    elif name_dataset == "ukb":
        all_modalities = {
            "nmr": 249,
            "ehr": 3584,
            "olink": 1463,
            "prs": 135,
            "bloodbio": 30,
            "baselinechars": 28,
            "localenvironment": 33,
            "arterialstiffness": 9,
            "anthropometry": 43,
            "bloodpressure": 12,
            "ecgduringexercise": 355,
            "eyemeasures": 310,
            "bonedensitometry": 27,
            "handgripstrength": 2,
            "spirometry": 29,
            "touchscreen": 119,
            "cognitivefunction": 27,
            "hearingtest": 68,
            "verbalinterview": 224,
            "bloodcount": 31,
            "urineassays": 4,
            "telomeres": 4,
            "infectiousdiseases": 66,
        }
        n_modalities = len(all_modalities)
        if n_modalities == 0:
            raise ValueError("No modalities found in dataloader output.")
        x = []
        for mod in all_modalities.keys():
            if mod in dataloader_output and "tabular_data" in dataloader_output[mod]:
                x.append(dataloader_output[mod]["tabular_data"])
        """ 
        x = [
            dataloader_output["nmr"]["tabular_data"],
            dataloader_output["ehr"]["tabular_data"],
            dataloader_output["olink"]["tabular_data"],
            dataloader_output["prs"]["tabular_data"],
            dataloader_output["bloodbio"]["tabular_data"],
            dataloader_output["baselinechars"]["tabular_data"],
            dataloader_output["localenvironment"]["tabular_data"],
            dataloader_output["arterialstiffness"]["tabular_data"],
            dataloader_output["anthropometry"]["tabular_data"],
            dataloader_output["bloodpressure"]["tabular_data"],
            dataloader_output["ecgduringexercise"]["tabular_data"],
            dataloader_output["eyemeasures"]["tabular_data"],
            dataloader_output["bonedensitometry"]["tabular_data"],
            dataloader_output["handgripstrength"]["tabular_data"],
            dataloader_output["spirometry"]["tabular_data"],
            dataloader_output["touchscreen"]["tabular_data"],
            dataloader_output["cognitivefunction"]["tabular_data"],
            dataloader_output["hearingtest"]["tabular_data"],
            dataloader_output["verbalinterview"]["tabular_data"],
            dataloader_output["bloodcount"]["tabular_data"],
            dataloader_output["urineassays"]["tabular_data"],
            dataloader_output["telomeres"]["tabular_data"],
            dataloader_output["infectiousdiseases"]["tabular_data"],
        ]
        """
    elif name_dataset == "mystery_mml":
        x = [dataloader_output["x_m1"], dataloader_output["x_m2"]]
    else: 
        raise NotImplementedError

    return x

def get_target(name_dataset, dataloader_output):
    if name_dataset == "mimic_haim" or name_dataset == "mimic_symile":
        y = dataloader_output["target"]
    elif name_dataset == "mosi" or name_dataset == "mosei":
        y = dataloader_output["label"]
    elif name_dataset == "crema_d":
        y = dataloader_output["target"]
    elif name_dataset == "ch_sims" or name_dataset == "ch_sims_v2":
        y = dataloader_output["label"]
    elif name_dataset == "inspect":
        y = dataloader_output["target"]
    elif name_dataset == "ukb":
        y = dataloader_output["labels"]["tabular_data"]
    elif name_dataset == "mystery_mml":
        y = dataloader_output["label"]
    else:
        raise NotImplementedError
    return y