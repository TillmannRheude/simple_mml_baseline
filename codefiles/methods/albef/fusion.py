import torch
import torch.nn as nn

class IncrementalCrossAttentionFusion(nn.Module):
    """
    A multimodal fusion module that performs symmetric, incremental cross-attention,
    inspired by the original ALBEF architecture.

    - A separate fusion encoder is created for each modality.
    - At each layer of a modality's fusion encoder, it first performs self-attention
      on its own sequence and then performs cross-attention to the other modalities.
    - Finally, the output [CLS] tokens from all fusion paths are combined.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        num_modalities: int = 2, 
    ):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.fusion_encoders = nn.ModuleList(
            [
                nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
                for _ in range(num_modalities)
            ]
        )

        self.final_fusion_layer = nn.TransformerEncoderLayer(
             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
             dropout=dropout, batch_first=True,
        )

    def forward(self, modality_embeddings: list):
        """
        Args:
            modality_embeddings (list): A list of tensors [B, S, D] from unimodal encoders.
        
        Returns:
            torch.Tensor: A fused representation of shape [B, D].
        """
        num_modalities = len(modality_embeddings)
        fused_cls_tokens = []

        for i in range(num_modalities):
            # The modality's own sequence is the target for the decoder
            query_sequence = modality_embeddings[i]
            
            # All other modalities are concatenated to form the memory
            context_sequences = [
                emb for j, emb in enumerate(modality_embeddings) if i != j
            ]
            
            if not context_sequences:
                # If only one modality, it can only self-attend.
                memory = query_sequence
            else:
                memory = torch.cat(context_sequences, dim=1)

            # Pass through the modality-specific fusion encoder
            # The TransformerDecoder will incrementally fuse `memory` into `query_sequence`
            fused_sequence = self.fusion_encoders[i](tgt=query_sequence, memory=memory)
            
            # Collect the fused CLS token
            fused_cls_tokens.append(fused_sequence[:, 0, :].unsqueeze(1))
        
        # Concatenate all the fused CLS tokens and pass through a final fusion layer
        all_fused_cls = torch.cat(fused_cls_tokens, dim=1)
        final_fused_representation = self.final_fusion_layer(all_fused_cls)
        
        # Return the mean of the final fused CLS tokens
        return final_fused_representation.mean(dim=1)


class PairwiseCrossAttentionFusion(nn.Module):
    """
    A multimodal fusion module that performs symmetric pairwise fusion.
    - For full sequences, each modality cross-attends to all other modalities.
    - For CLS tokens only, it falls back to self-attention on the concatenated tokens.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attention_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.cross_attention_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.final_fusion_layer = nn.TransformerEncoderLayer(
             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
             dropout=dropout, batch_first=True,
        )

    def forward(self, modality_embeddings: list):
        """
        Args:
            modality_embeddings (list): A list of tensors from unimodal encoders.
                                      Tensors can be [B, 1, D] for CLS tokens or
                                      [B, S, D] for full sequences.
        
        Returns:
            torch.Tensor: A fused representation of shape [B, D].
        """
        is_late_fusion = all(x.shape[1] == 1 for x in modality_embeddings)

        if is_late_fusion:
            concatenated_cls = torch.cat(modality_embeddings, dim=1)
            fused_representation = self.self_attention_encoder(concatenated_cls)
            return fused_representation.mean(dim=1)

        else:
            num_modalities = len(modality_embeddings)
            fused_cls_tokens = []

            for i in range(num_modalities):
                query_sequence = modality_embeddings[i]
                
                context_sequences = [
                    emb for j, emb in enumerate(modality_embeddings) if i != j
                ]
                
                if not context_sequences:
                    memory = query_sequence # Only one modality, self-attend
                else:
                    memory = torch.cat(context_sequences, dim=1)

                output_sequence = query_sequence
                for layer in self.cross_attention_layers:
                    output_sequence = layer(tgt=output_sequence, memory=memory)
                
                fused_cls_tokens.append(output_sequence[:, 0, :].unsqueeze(1))
            
            all_fused_cls = torch.cat(fused_cls_tokens, dim=1)
            final_fused_representation = self.final_fusion_layer(all_fused_cls)
            
            return final_fused_representation.mean(dim=1)
