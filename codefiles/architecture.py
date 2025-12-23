import torch 
import torch.nn as nn 
import torch.nn.functional as F

from codefiles.encoders import Encoders
from codefiles.transformer import Multimodal_Transformer

class Multimodal_Architecture(nn.Module):

    def __init__(
            self, 
            encoders: Encoders = Encoders(),
            transformer: Multimodal_Transformer = Multimodal_Transformer(),
            full_seq: bool = False,
            unimodal: str = None  # unimodal_0, unimodal_1, unimodal_2, ...
    ) -> None: 
        super().__init__()

        self.encoders = encoders
        self.transformer = transformer
        self.full_seq = full_seq
        self.unimodal = unimodal
        
    def get_src_mask(
            self,
            x: list = []
    ) -> torch.Tensor:
        bs = x[0].shape[0]

        # create src_mask indicating which modalities are missing
        src_mask = [torch.isnan(x[i]).contiguous().view(bs, -1).all(1) for i in range(len(x))]  # any(1) --> all(1)
        src_mask = torch.stack(src_mask, dim=-1)
        # True stands for missing modality and False stands for available modality
        assert src_mask.ndim == 2, "src_mask should have shape (batch_size, num_modalities), i.e., 2 dimensions."
        return src_mask
    
    def _unimodal_probing(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        src_mask = self.get_src_mask(x)
        num_modalities = src_mask.shape[1]

        # due to computational reasons, replace NaNs with 0s (but mask them)
        x = [torch.nan_to_num(x[i], nan=0.0) for i in range(len(x))]
        # multimodal pipeline
        x_embs = self.encoders(x, epoch=None, steps_per_epoch=None, full_seq=self.full_seq)

        # convert src_mask to seq len if full seq
        if self.full_seq:
            seq_lens = [curr_x_emb.shape[1] for curr_x_emb in x_embs]
            # split src_mask to list of src_masks for each modality
            src_masks = torch.split(src_mask, 1, dim=1)
            src_mask = [curr_src_mask.repeat(1, seq_len) for curr_src_mask, seq_len in zip(src_masks, seq_lens)]

            src_mask = torch.cat(src_mask, dim=1)
            x_embs = torch.cat(x_embs, dim=1)

        casestudy_logits = []
        for m in range(num_modalities):
            src_mask = torch.ones_like(src_mask, dtype=torch.bool)
            src_mask[:, m] = False
            
            x = self.transformer(x_embs, src_mask, y)["logits"]

            casestudy_logits.append(x)
        casestudy_logits = torch.stack(casestudy_logits, dim=1)

        return casestudy_logits

    def _unimodal_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # get modality -> unimodal_0 -> 0 
        n_modality = int(self.unimodal.split("_")[1])
        modality = x[n_modality]

        # fill nans with zeros in modality
        modality = torch.nan_to_num(modality, nan=0.0)
        
        # pass modality through encoder
        modality_emb = self.encoders.encoders[n_modality](modality)

        # pass modality through transformer
        x = self.transformer(modality_emb, src_mask=None, y=y)

        return x 

    def forward(
            self, 
            x: list = [],
            y: torch.Tensor = None,
            epoch: int = None,
            steps_per_epoch = None,
            return_details: bool = False
    ) -> torch.Tensor:
        if self.unimodal is not None and self.unimodal != "None" and "intersect" not in self.unimodal:
            return self._unimodal_forward(x, y)
        else:
            bs = x[0].shape[0]

            # get src mask for missing modality values
            src_mask = self.get_src_mask(x)

            # due to computational reasons, replace NaNs with 0s (but mask them)
            x = [torch.nan_to_num(x[i], nan=0.0) for i in range(len(x))]
            # multimodal pipeline
            x_embs = self.encoders(x, epoch=epoch, steps_per_epoch=steps_per_epoch, full_seq=self.full_seq)

            # convert src_mask to seq len if full seq
            if self.full_seq:
                seq_lens = [curr_x_emb.shape[1] for curr_x_emb in x_embs]
                # split src_mask to list of src_masks for each modality
                src_masks = torch.split(src_mask, 1, dim=1)
                src_mask = [curr_src_mask.repeat(1, seq_len) for curr_src_mask, seq_len in zip(src_masks, seq_lens)]
            
            x = self.transformer(x_embs, src_mask, y)  # .squeeze()
            
            if return_details:
                return x["shared_embeddings_g"], src_mask
            else:
                return x