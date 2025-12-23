import torch 
import torch.nn as nn 

from torch.nn import functional as F

from codefiles.encoders import AddCLSToken, ExtractCLSToken, AddPE
from codefiles.losses.nanbce import WeightedNaNBCEWithLogitsLoss


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

class SqueezeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze()

class AUG_Transformer(nn.Module):

    """
    https://github.com/njustkmg/NeurIPS25-AUG
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_layers: int = 4,
        dim_output: int = 10,
        num_modalities: int = 2,
        task_type: str = "bce",
        merge_alphas: list[float] = [0.5, 0.5],
        lambda_smooth: float = 0.5,
    ) -> None: 
        super().__init__()

        self.num_modalities = num_modalities
        self.dim_output = dim_output
        self.d_model = d_model
        self.merge_alphas = merge_alphas
        self.task_type = task_type
        self.lambda_smooth = lambda_smooth

        self.opt = None 

        if task_type == "bce":
            self.criterion = WeightedNaNBCEWithLogitsLoss(reduction="none")
            self.activation = nn.Sigmoid()
            self.f_activation = F.sigmoid
        elif task_type == "ce":
            self.criterion = nn.CrossEntropyLoss(reduction="none")
            self.activation = nn.Softmax(dim=-1)
            self.f_activation = F.softmax
        else:
            raise ValueError(f"Invalid task type: {task_type}")

        self.embedding_modality = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
            for _ in range(num_modalities)
        ])
        self.additional_layers_modality = nn.ModuleList([
            nn.ModuleList([]) for _ in range(num_modalities)
        ])

        self.t_loss = Averager()
        self.t_modalities = [Averager() for _ in range(num_modalities)]
        self.score_modality = [0.0 for _ in range(num_modalities)]

        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(d_model, dim_output)
        self.apply(self._init_weights)

        # Transformer Head 
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers
        )
        self.transformer_cls = nn.ModuleList([
            AddCLSToken(d_model),
            AddPE(d_model),
            self.transformer,
            ExtractCLSToken(),
            self.linear_out,
            # SqueezeLayer()
        ])

    def set_opt(self, opt: torch.optim.Optimizer):
        self.opt = opt
    
    def reset_scores_and_t_stats(self):
        self.t_loss = Averager()
        self.t_modalities = [Averager() for _ in range(self.num_modalities)]
        self.score_modality = [0.0 for _ in range(self.num_modalities)]
    
    def calculate_performance_ratios(self):
        total_score_sum = sum(self.score_modality)
        performance_ratios = [self.score_modality[i] / total_score_sum for i in range(self.num_modalities)]

        return performance_ratios
    
    def add_layer(self, idx_modality: int):
        device = next(self.parameters()).device
        new_layer = nn.Linear(self.d_model, self.d_model).to(device)
        new_layer.apply(self._init_weights)
        self.additional_layers_modality[idx_modality].append(new_layer)

    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
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

    def classfier(self, feature_modality, idx_modality: int):
        result = self.embedding_modality[idx_modality](feature_modality)
        feature = self.head(result)
        o_fea = feature
        add_fea = None
        i = 0
        layerlen = len(self.additional_layers_modality[idx_modality])
        for layer in self.additional_layers_modality[idx_modality]:
            addf = self.relu(layer(feature_modality))
            add_fea = self.head(addf)
            feature = feature + add_fea
            i=i+1
            if i < layerlen:
                o_fea = feature

        return feature, o_fea, add_fea

    def get_batch_score(self,out, y):
        if self.task_type == 'ce':
            # Original Code: Uses Softmax and selects the probability of the argmax index
            return sum([self.f_activation(out, dim=-1)[i][torch.argmax(y[i])] for i in range(out.size(0))])

        elif self.task_type == 'bce':
            # Analogy: Uses Sigmoid. If y is 1, take p. If y is 0, take (1-p).
            probs = torch.sigmoid(out)
            # Create a mask for valid targets (not NaN)
            mask = ~torch.isnan(y)
            # Replace NaNs in y with 0 for the comparison (masked out later)
            y_clean = y.nan_to_num(0)
            
            # Calculate individual scores
            scores = torch.where(y_clean > 0.5, probs, 1.0 - probs)
            
            # Only sum up scores where the target was valid
            return (scores * mask).sum().item()
            
    def head(self, x):
        for layer in self.transformer_cls:
            x = layer(x)
        return x
    
    def convert_y_chsims(self, y: torch.Tensor) -> torch.Tensor:
        five_classmapping = {
            -1.0: 0, -0.8: 0,
            -0.6: 1, -0.4: 1, -0.2: 1, -0.1: 1,
            0.0: 2,
            0.2: 3, 0.4: 3, 0.6: 3,
            0.8: 4, 1.0: 4
        }  # 0=negative, 1=weakly negative, 2=neutral, 3=weakly positive, 4=positive
        y = torch.tensor([five_classmapping.get(float(v), -1) for v in y], device=y.device, dtype=torch.long)
        return y

    def forward(
        self, 
        x: torch.Tensor = torch.Tensor,
        src_mask: torch.Tensor = torch.Tensor,
        y: torch.Tensor = None,
    ) -> dict:

        # y = F.one_hot(y.squeeze(), num_classes=6).float()
        # Only apply one_hot if y is not already a float tensor (i.e. it's class indices)
        # Only one-hot encode if we have >1 output classes.
        if (y.dim() == 1 or (y.dim() == 2 and y.shape[1] == 1)) and self.dim_output > 1:
            # check if y has negative values 
            if torch.any(y < 0):
                y = self.convert_y_chsims(y)
            y = F.one_hot(y.squeeze().long(), num_classes=self.dim_output).float()
        # Ensure y is float for BCE
        y = y.float()
        # Ensure shape matches logits [Batch, 1] for single-label tasks
        if self.dim_output == 1 and y.dim() == 1:
            y = y.unsqueeze(1)

        features_modality = torch.split(x, 1, dim=1)

        loss_modality_list = []
        out_modality_list = []
        for idx_modality, feature_modality in enumerate(features_modality):
            out_modality, o_fea, add_fea = self.classfier(feature_modality, idx_modality)

            if add_fea is None:
                loss_modality = self.criterion(out_modality, y).mean()
            else:
                kl = y*self.activation(o_fea.detach())
                loss_modality = self.criterion(out_modality, y).mean() + self.criterion(o_fea, y).mean() + self.criterion(add_fea, y).mean() - self.lambda_smooth * self.criterion(add_fea, kl).mean() 

            loss_modality_list.append(loss_modality)
            out_modality_list.append(out_modality)
        
        loss = torch.tensor(0.0, device=x.device)
        for i, loss_modality in enumerate(loss_modality_list):
            loss = loss + loss_modality
        
        with torch.no_grad():
            self.t_loss.add(loss.item())
            for i, loss_modality in enumerate(loss_modality_list):
                self.t_modalities[i].add(loss_modality.item())

            for i, out_modality in enumerate(out_modality_list):
                tmp_modality = self.get_batch_score(out_modality, y)
                self.score_modality[i] += tmp_modality

        # out = merge_alpha * out_a + (1 - merge_alpha) * out_v
        # merge alphas are only used for logging and logits in the official code repository. 
        logits = sum([out_modality * self.merge_alphas[i] for i, out_modality in enumerate(out_modality_list)])
        
        return {
            "unimodal_logits": out_modality_list,
            "logits": logits, 
            "loss": loss,
            "aug": True
        }
