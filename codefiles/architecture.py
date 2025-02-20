import torch 
import torch.nn as nn 

from codefiles.encoders import Encoders
from codefiles.transformer import Multimodal_Transformer

class Multimodal_Architecture(nn.Module):

    def __init__(
            self, 
            encoders: Encoders = Encoders(),
            transformer: Multimodal_Transformer = Multimodal_Transformer()
    ) -> None: 
        super().__init__()

        self.encoders = encoders
        self.transformer = transformer

    def forward(
            self, 
            x: list = []
    ) -> torch.Tensor:
        bs = x[0].shape[0]

        # create src_mask indicating which modalities are missing
        src_mask = [torch.isnan(x[i]).view(bs, -1).any(1) for i in range(len(x))]
        src_mask = torch.stack(src_mask, dim=-1)
        assert src_mask.ndim == 2, "src_mask should have shape (batch_size, num_modalities), i.e., 2 dimensions."

        # due to computational reasons, replace NaNs with 0s (but mask them)
        x = [torch.nan_to_num(x[i], nan=0.0) for i in range(len(x))]
        
        # multimodal pipeline
        x = self.encoders(x)
        x = self.transformer(x, src_mask).squeeze()
        
        return x