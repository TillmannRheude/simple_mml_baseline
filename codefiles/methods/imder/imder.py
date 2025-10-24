import torch 
import torch.nn as nn 

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.methods.imder.dit import Diffusion_Transformer
from codefiles.methods.imder.diffusion import GaussianDiffusion1D

from codefiles.helpers import is_running_in_notebook  # for reloading modules instead of restarting kernel
if is_running_in_notebook():
    from codefiles.methods.imder import dit
    import importlib
    importlib.reload(dit)
from codefiles.methods.imder.dit import Diffusion_Transformer


class IMDer(nn.Module):

    """
    https://proceedings.neurips.cc/paper_files/paper/2023/file/372cb7805eaccb2b7eed641271a30eec-Paper-Conference.pdf

    https://github.com/mdswyz/IMDer
    """

    def __init__(
        self,
        params_transformerhead: dict = {
            "d_model": 512,
            "nhead": 4,
            "dim_feedforward": 1024,
            "dropout": 0.0,
            "num_layers": 4,
            "dim_output": 10,
        },
        params_ddpm: dict = {
            "sampling_iter": 10,
            "n_steps": 10,
            "n_modalities": 2,
            "hidden_dim": 512,
        },
        params_imder: dict = {
            "beta": 0.1,
        }
    ) -> None:
        super().__init__()
        self.si = params_ddpm["sampling_iter"]
        self.n_modalities = params_ddpm["n_modalities"]
        self.beta = params_imder["beta"]

        # Conditioning Fusion
        self.condition_fusion = nn.Conv1d(
            in_channels=params_ddpm["d_model"],
            out_channels=params_ddpm["d_model"],
            kernel_size=1
        )

        # Refinement Decoders
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(params_transformerhead["d_model"], params_transformerhead["d_model"] * 2),
                nn.GELU(),
                nn.Linear(params_transformerhead["d_model"] * 2, params_transformerhead["d_model"])
            ) for _ in range(self.n_modalities)
        ])

        # Projection Head
        self.linear_out = nn.Linear(params_transformerhead["d_model"], params_transformerhead["dim_output"])
        self.apply(self._init_weights)

        # DDPMs
        dits = self._init_scoremodels(params_ddpm)
        self.ddpms = nn.ModuleList([
            GaussianDiffusion1D(
                model=dits[i],
                timesteps=params_ddpm["n_steps"]
            ) for i in range(params_ddpm["n_modalities"])]
        )

        # Transformer Head 
        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=params_transformerhead["d_model"], 
                nhead=params_transformerhead["nhead"], 
                dim_feedforward=params_transformerhead["dim_feedforward"], 
                dropout=params_transformerhead["dropout"],
                batch_first=True,
            ),
            num_layers=params_transformerhead["num_layers"]
        )
        self.transformer = nn.ModuleList([
            AddCLSToken(params_transformerhead["d_model"]),
            AddPE(params_transformerhead["d_model"]),
            transformer,
            ExtractCLSToken(),
            self.linear_out
        ])
        
    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Parameter):
            if m.dim() > 1:
                torch.nn.init.xavier_normal_(m)
            else:
                torch.nn.init.zeros
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def _init_scoremodels(self, params_ddpm: dict = {}):
        scoremodels = nn.ModuleList([
            Diffusion_Transformer(
                input_dim=params_ddpm["d_model"],
                output_dim=params_ddpm["d_model"],
                hidden_dim=params_ddpm["hidden_dim"],
                num_layers=params_ddpm["num_layers"],
                dropout=params_ddpm["dropout"],
                nhead=params_ddpm["nhead"],
                condition_dim=params_ddpm["d_model"])
                for _ in range(params_ddpm["n_modalities"])
        ])
        return scoremodels

    def _add_cls_token_mask_to_src_mask(
            self, 
            src_mask: torch.Tensor
    ) -> torch.Tensor:
        assert src_mask.dtype == torch.bool
        src_mask = torch.cat(
            [
                torch.zeros(src_mask.shape[0], 1, dtype=torch.bool, device=src_mask.device), 
                src_mask
            ], dim=1
        ).to(dtype=torch.bool)
        return src_mask 

    def _train_ddpms(
        self, 
        condition: torch.Tensor, 
        src_mask: torch.Tensor, 
    ) -> list:
        # get shapes
        bs, seq_len, emb_dim = condition.shape

        # Precompute modality boundaries
        self.modality_indices = torch.cumsum(torch.tensor([0] + self.seq_lens, device=src_mask.device), dim=0)

        ddpm_losses = []
        for curr_ddpm in range(self.n_modalities):
            target_seq_len = self.seq_lens[curr_ddpm]
            si_squeezed_shape_target = (bs * self.si, target_seq_len, emb_dim)
            cond_seq_len = seq_len - target_seq_len
            si_squeezed_shape_cond = (bs * self.si, cond_seq_len, emb_dim)

            # Get target and condition embeddings and masks
            start_idx = self.modality_indices[curr_ddpm]
            end_idx = self.modality_indices[curr_ddpm + 1]
            embtarget = condition[:, start_idx:end_idx, :]
            target_mask = src_mask[:, start_idx:end_idx]

            cond_parts = []
            cond_mask_parts = []
            if curr_ddpm > 0:
                cond_parts.append(condition[:, :start_idx, :])
                cond_mask_parts.append(src_mask[:, :start_idx])
            if curr_ddpm < self.n_modalities - 1:
                cond_parts.append(condition[:, end_idx:, :])
                cond_mask_parts.append(src_mask[:, end_idx:])
            curr_conds = torch.cat(cond_parts, dim=1)
            cond_mask = torch.cat(cond_mask_parts, dim=1)
            
            # Create loss mask: train if target is missing and at least one condition is present
            target_is_missing = target_mask.all(dim=1)
            at_least_one_cond_present = ~cond_mask.all(dim=1) if cond_seq_len > 0 else torch.zeros(bs, device=src_mask.device, dtype=torch.bool)
            loss_mask_per_sample = target_is_missing & at_least_one_cond_present

            # Prepare conditioning mask for the DDPM
            src_mask_scoremodel = cond_mask.repeat(self.si, 1)

            # Handle higher sampling iterations+
            curr_conds = curr_conds[:, None, ...].repeat(1, self.si, 1, 1).view(si_squeezed_shape_cond) 
            embtarget = embtarget[:, None, ...].repeat(1, self.si, 1, 1).view(si_squeezed_shape_target)

            # Prepare final loss mask
            loss_mask = loss_mask_per_sample[:, None].repeat(1, self.si).view(bs * self.si)
            mask = loss_mask[..., None, None].repeat(1, target_seq_len, emb_dim)

            # Calculate loss
            ddpm_loss = self.ddpms[curr_ddpm](embedding=embtarget, condition=curr_conds, src_mask=src_mask_scoremodel)["loss"]            
            loss = ddpm_loss * mask
            if mask.sum() > 0:
                ddpm_loss = loss.sum() / mask.sum()
            else:
                ddpm_loss = torch.tensor(0.0, device=loss.device)

            ddpm_losses.append(ddpm_loss)

        return ddpm_losses

    def _sample_ddpms(
        self, 
        condition: torch.Tensor, 
        src_mask: torch.Tensor, 
    ) -> list:
        # get shapes
        bs, seq_len, emb_dim = condition.shape
        si_squeezed_shape_cond = (bs * self.si, -1, emb_dim)

        if not hasattr(self, 'modality_indices'):
            self.modality_indices = torch.cumsum(torch.tensor([0] + self.seq_lens, device=src_mask.device), dim=0)

        ddpm_outputs = []
        for curr_ddpm in range(self.n_modalities):
            target_seq_len = self.seq_lens[curr_ddpm]
            sampling_shape = (bs * self.si, target_seq_len, emb_dim)

            # Get condition embeddings and mask
            start_idx = self.modality_indices[curr_ddpm]
            end_idx = self.modality_indices[curr_ddpm + 1]

            cond_parts = []
            cond_mask_parts = []
            if curr_ddpm > 0:
                cond_parts.append(condition[:, :start_idx, :])
                cond_mask_parts.append(src_mask[:, :start_idx])
            if curr_ddpm < self.n_modalities - 1:
                cond_parts.append(condition[:, end_idx:, :])
                cond_mask_parts.append(src_mask[:, end_idx:])
            
            if cond_parts:
                curr_conds = torch.cat(cond_parts, dim=1)
                cond_mask = torch.cat(cond_mask_parts, dim=1)
            else:
                curr_conds = torch.empty(bs, 0, emb_dim, device=condition.device, dtype=condition.dtype)
                cond_mask = torch.empty(bs, 0, device=src_mask.device, dtype=torch.bool)

            # Create conditioning mask for the DDPM
            src_mask_scoremodel = cond_mask.repeat(self.si, 1)

            # Sampling
            curr_conds = curr_conds[:, None, ...].repeat(1, self.si, 1, 1).view(si_squeezed_shape_cond)

            ddpm_output = self.ddpms[curr_ddpm].p_sample_loop(
                shape=sampling_shape,
                condition=curr_conds,
                src_mask=src_mask_scoremodel
            ).view(bs, self.si, target_seq_len, emb_dim)
            
            ddpm_outputs.append(ddpm_output)
        
        return ddpm_outputs
    
    def _choose_modalities(
        self,
        x: torch.Tensor, 
        x_imputed: list = [torch.Tensor, torch.Tensor],
        src_mask: torch.Tensor = torch.Tensor
    ) -> list:
        modalities_ = []
        for i in range(len(x_imputed)):
            bs, seq_len, embdim = x.shape
            seq_len = 1

            mask = src_mask[:, i][:, None, None, None].repeat(1, self.si, seq_len, embdim)

            curr_mod_emb = torch.where(
                mask, 
                x_imputed[i], 
                x[:, i, :][:, None, None, ...].repeat(1, self.si, 1, 1)
            )
            curr_mod_emb = curr_mod_emb.view(bs, self.si * seq_len, embdim)
            modalities_.append(curr_mod_emb)

        chosen_modality_embeddings = torch.cat(modalities_, dim=1)

        return chosen_modality_embeddings
    
    def _create_final_src_mask(
        self, 
        src_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len = 1

        # Create mask that's False for sampled embeddings and first sequence of original, True for repeated sequences
        src_masks_per_mod = []
        for i in range(self.n_modalities):
            mod_mask = torch.where(src_mask[:, i][:, None].repeat(1, self.si * seq_len), False, True)  # Don't mask sampled embeddings
            mod_mask[:, :seq_len] = False  # Don't mask first sequence of original embeddings
            src_masks_per_mod.append(mod_mask)
        src_mask = torch.cat(src_masks_per_mod, dim=1)

        return src_mask

    def forward_w_imputations(
        self,
        x: torch.Tensor = torch.Tensor,
        src_mask: torch.Tensor = torch.Tensor
    ) -> torch.Tensor:

        # DDPMs: Train / Sample
        #x_imputed = self._sample_ddpms(x, src_mask)

        # Create Transformer input, i.e., original modality if available, else imputed modality
        #x = self._choose_modalities(x, x_imputed, src_mask)

        # Transformer Head
        for layer in self.transformer:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:                
                x = layer(
                    x, 
                    src_key_padding_mask=None  # use imputations instead
                )
            else:
                x = layer(x)

        return x.squeeze()

    def forward(
        self,
        x: torch.Tensor = torch.Tensor,
        src_mask: torch.Tensor = torch.Tensor,
        y: torch.Tensor = None
        ) -> dict:

        self.seq_lens = [sm.shape[1] for sm in src_mask]

        if isinstance(x, list):
            x = torch.cat(x, dim=1)
            src_mask = torch.cat(src_mask, dim=1)

        # DDPMs: Train / Sample
        ddpm_criterions = self._train_ddpms(x, src_mask)
        ddpm_loss = torch.mean(torch.stack(ddpm_criterions))
        x_imputed_raw = self._sample_ddpms(x, src_mask)

        # Refine samples with decoders and compute reconstruction loss
        x_imputed_refined = []
        for i, decoder in enumerate(self.decoders):
            # x_imputed_raw[i] has shape [bs, si, 1, emb_dim]
            refined = decoder(x_imputed_raw[i])
            x_imputed_refined.append(refined)

        # Average over sampling iterations for a single prediction
        x_imputed_refined_mean = [t.mean(dim=1) for t in x_imputed_refined]
        x_imputed_final = torch.cat(x_imputed_refined_mean, dim=1) # [bs, sum(seq_lens), emb_dim]

        # Reconstruction Loss (Lrec)
        loss_recon = nn.functional.mse_loss(x_imputed_final, x, reduction="none")
        mask_recon = src_mask.unsqueeze(-1).expand_as(loss_recon)
        if mask_recon.sum() > 0:
            reconstruction_loss = loss_recon[mask_recon].mean()
        else:
            reconstruction_loss = torch.tensor(0.0, device=x.device)

        imputation_loss = ddpm_loss + reconstruction_loss
        imputation_loss = self.beta * imputation_loss

        # Create Transformer input: original modality if available, else refined imputed modality
        x_complete = torch.where(
            src_mask.unsqueeze(-1).expand_as(x),
            x_imputed_final,
            x
        )
        # src_mask = self._add_cls_token_mask_to_src_mask(torch.zeros_like(src_mask, dtype=torch.bool)) # All modalities are present now

        # Transformer Head 
        for layer in self.transformer:
            if isinstance(layer, nn.TransformerEncoder) and src_mask is not None:                
                x_complete = layer(
                    x_complete,
                    # src_key_padding_mask=src_mask
                )
            else:
                x_complete = layer(x_complete)

        return {
            "logits": x_complete,
            "losses": {
                "imputation_loss": imputation_loss
            }
        }
