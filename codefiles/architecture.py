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
            full_seq: bool = False
    ) -> None: 
        super().__init__()

        self.encoders = encoders
        self.transformer = transformer
        self.full_seq = full_seq

    def get_src_mask(
            self,
            x: list = []
    ) -> torch.Tensor:
        bs = x[0].shape[0]

        # create src_mask indicating which modalities are missing
        src_mask = [torch.isnan(x[i]).view(bs, -1).any(1) for i in range(len(x))]
        src_mask = torch.stack(src_mask, dim=-1)
        # True stands for missing modality and False stands for available modality
        assert src_mask.ndim == 2, "src_mask should have shape (batch_size, num_modalities), i.e., 2 dimensions."
        return src_mask

    def forward(
            self, 
            x: list = [],
            y: torch.Tensor = None,
            epoch: int = None,
            steps_per_epoch = None,
            return_details: bool = False
    ) -> torch.Tensor:
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
    
class Multimodal_CEN_Architecture(Multimodal_Architecture):

    def __init__(
            self, 
            encoders: Encoders = Encoders(),
            transformer: Multimodal_Transformer = Multimodal_Transformer()
    ) -> None: 
        super().__init__(encoders=encoders, transformer=transformer)

        self.num_modalities = 2

        # convert encoders to nn.ModuleList
        for i, enc in enumerate(encoders.encoders):
            encoders.encoders[i].encoder = nn.ModuleList(enc.encoder)

        self.encoders = ChannelExchangeNetwork(encoders=encoders)
        self.transformer = transformer

        # Create a learnable parameter vector, one weight per modality
        self.alpha = nn.Parameter(torch.ones(self.num_modalities, requires_grad=True))
    
    def forward(
            self, 
            x: list = []
    ) -> torch.Tensor:
        bs = x[0].shape[0]

        # get src mask for missing modality values
        src_mask = self.get_src_mask(x)
        # due to computational reasons, replace NaNs with 0s (but mask them)
        x = [torch.nan_to_num(x[i], nan=0.0) for i in range(len(x))]
        # multimodal pipeline
        x, l1_loss = self.encoders(x)

        # alpha ensemble from CEN paper (https://github.com/yikaiw/CEN/blob/master/semantic_segmentation/models/model.py#L314)
        #alpha_soft = F.softmax(self.alpha, dim=0) # Normalize alpha weights -> [num_modalities]
        #ensemble_output = sum(alpha_soft[i] * x[i] for i in range(self.num_modalities))
        x = torch.stack(x, dim=1)
        x = self.transformer(x, src_mask).squeeze()
        
        return x, l1_loss
            
class ChannelExchangeNetwork(nn.Module):
    
    """ 
    Re-implementation of CEN from the paper.
    Y. Wang, W. Huang, F. Sun, T. Xu, Y. Rong, and J. Huang, “Deep Multimodal Fusion by Channel Exchanging”
    """
    
    def __init__(
            self, 
            encoders: Encoders = Encoders(),
            exchange_threshold: float = 2e-2
    ) -> None: 
        super().__init__()

        self.encoders = encoders
        self.exchange_threshold = exchange_threshold

    def channel_exchange(
            self,
            x: list = [torch.Tensor, torch.Tensor],
            gammas: list = [torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
        """
        Exchange channels where gamma is below the threshold.
        Gradients are detached from the replaced channel and back-propagated through the new ones.
        """
        n_mod = len(x)
        assert x[0].ndim == 4, "x should have shape (batch_size, num_channels, height, width), i.e., 4 dimensions."
        
        # Create copies of inputs to modify
        x_new = [tensor.clone() for tensor in x]
        
        # For every tensor in x: take mean of other tensors for indices of gamma where gamma < threshold
        for i_mod in range(n_mod):
            # Get indices of gamma where gamma < threshold
            idx = gammas[i_mod] < self.exchange_threshold
            
            print(f"Min gamma: {gammas[i_mod].min().item()}, Max gamma: {gammas[i_mod].max().item()}")
            
            # Original channels that will be replaced - detach their gradients
            original_channels = x[i_mod][:, idx, ...].detach()
            
            # Get mean of other modality tensors - keep gradients intact
            other_mods = [x[j] for j in range(n_mod) if j != i_mod]
            if other_mods:
                x_wo_curr_mod = torch.stack(other_mods, dim=0)
                mean_x_wo_curr_mod = torch.mean(x_wo_curr_mod, dim=0)
                
                # Replace channels using combination of detached original and gradient-preserving new data
                x_new[i_mod][:, idx, ...] = mean_x_wo_curr_mod[:, idx, ...]

        # Calculate l1 sparsity loss between all pairs of gammas
        total_l1_sparsity_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(n_mod):
            total_l1_sparsity_loss = total_l1_sparsity_loss + torch.sum(gammas[i].abs())  # Direct L1 regularization for sparsity
        
        return x_new, total_l1_sparsity_loss

    def forward(
            self, 
            x: list = []
    ) -> torch.Tensor:
        n_encoder_layers = len(self.encoders.encoders[0].encoder)
        n_mod = len(x)

        total_l1_loss = 0
        for i_enclayer in range(n_encoder_layers):
            gammas = []
            # apply encoder to each modality
            for i_mod in range(n_mod):
                curr_layer_curr_mod = self.encoders.encoders[i_mod].encoder[i_enclayer]
                x[i_mod] = curr_layer_curr_mod(x[i_mod])
                # if current layer is Conv2d ...
                if isinstance(curr_layer_curr_mod, nn.Conv2d):
                    # ... get gammas from next (bn) layer
                    next_layer_curr_mod = self.encoders.encoders[i_mod].encoder[i_enclayer+1]
                    curr_gammas = next_layer_curr_mod.weight.abs()
                    gammas.append(curr_gammas)

            # after all modality paths through i_layer: if current layer is Conv2d ...
            if isinstance(curr_layer_curr_mod, nn.Conv2d):     
                # ... apply channel exchange
                x, l1_loss = self.channel_exchange(
                    x=x, 
                    gammas=gammas
                )
                total_l1_loss += l1_loss
        
        return x, total_l1_loss