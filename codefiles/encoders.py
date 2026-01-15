import math
import torch 

import torch.nn as nn 
import torchvision.models as models

from transformers import (
    BertModel, BertConfig,
    VivitModel, VivitConfig, 
    Wav2Vec2Config, Wav2Vec2Model
)

from codefiles.methods.regbn.rbn_adjusted import RegBN

""" Helpers """
class Sequential_Tokenizer_and_Bert(nn.Module):
    def __init__(self, chinese: bool = False, full_seq: bool = False) -> None:
        super().__init__()
        self.use_finetune = False
        self.use_pretrain_bert = True
        self.full_seq = full_seq
        self.chinese = chinese

        if self.use_pretrain_bert:
            self.language_model = (
                BertModel.from_pretrained("bert-base-uncased")
                if not chinese
                else BertModel.from_pretrained(
                    "bert-base-chinese",
                    cache_dir="/path/to/data/tmp_hf_cache",
                )
            )
        else:
            self.language_model = BertModel(BertConfig())

        if not self.use_finetune:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def forward(self, text):
        if self.chinese:
            input_ids = text[:, 0, :].long()
            input_mask = text[:, 1, :].float()
            if self.use_finetune:
                last_hidden_states = self.language_model(input_ids=input_ids, attention_mask=input_mask)[0]
            else:
                with torch.no_grad():
                    last_hidden_states = self.language_model(input_ids=input_ids, attention_mask=input_mask)[0]
        else:
            input_ids = text[:, 0, :].long()
            input_mask = text[:, 1, :].float()
            segment_ids = text[:, 2, :].long()
            if self.use_finetune:
                last_hidden_states = self.language_model(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids
                )[0]
            else:
                with torch.no_grad():
                    last_hidden_states = self.language_model(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids
                )[0]

        return last_hidden_states if self.full_seq else last_hidden_states[:, 0, :]

class Unsqueeze_Sequence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, None, :]

class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 dropout: float = 0.0, 
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class AddCLSToken(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embdim) * 0.02)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

class ExtractCLSToken(nn.Module):
    def forward(self, x):
        return x[:, 0, :]

class AddPE(nn.Module):
    def __init__(self, embdim, max_len=7000):
        super().__init__()
        self.pe = PositionalEncoding(d_model=embdim, max_len=max_len)
        
    def forward(self, x):
        return self.pe(x.permute(1, 0, 2)).permute(1, 0, 2)


""" Encoders """
class Parent_Encoder(nn.Module):
    
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = None
        self.encoder_full_seq = None

        self.apply(self._init_weights)

    def _init_weights(
            self,
            m
        ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(
            self,
            x: torch.Tensor,
            full_seq: bool = False
    ) -> torch.Tensor:
        if full_seq:
            return self.encoder_full_seq(x)
        else:
            return self.encoder(x)[:, None, :]
    
class Encoders(nn.Module):

    def __init__(
            self, 
            encoders: nn.ModuleList = nn.ModuleList([]),
    ) -> None: 
        super().__init__()

        self.encoders = encoders
        
    def forward(
            self, 
            x: list = [
                torch.zeros(128, 249), 
                torch.zeros(128, 249)
            ],
            epoch: int = None,
            steps_per_epoch: int = None,
            full_seq: bool = False
    ) -> torch.Tensor:
        n_mod = len(x)
        
        common_latents = []
        total_l1_loss = 0
        for i in range(n_mod):
            encoder_out = self.encoders[i](x[i], full_seq=full_seq)
            # if encoder_out is tuple, take the first element
            if isinstance(encoder_out, tuple):
                encoder_out, l1_loss = encoder_out
                total_l1_loss += l1_loss
            common_latents.append(encoder_out)
        
        if full_seq:
            return common_latents
        else:
            if total_l1_loss > 0:
                return torch.cat(common_latents, dim=1), total_l1_loss
            else:
                return torch.cat(common_latents, dim=1)
        
class Encoders_RegBN(nn.Module):

    """
    https://arxiv.org/abs/2310.00641

    https://github.com/mogvision/regbn 
    """

    def __init__(
            self, 
            encoders: nn.ModuleList = nn.ModuleList([]),
            output_dim: int = 512,
            regbn_params: dict = {
                "momentum": 0.02,
                "sigma_THR": 0.0,
                "sigma_MIN": 0.0,
                "normalize_input": False,
                "normalize_output": False,
                "affine": False,
                "beta1": 0.9,
                "beta2": 0.99,
            },
            n_modalities: int = 2,
            reference_modality_idx: int = -1
    ) -> None: 
        super().__init__()
        
        self.n_modalities = n_modalities
        if self.n_modalities > 1:
            # Create a list of RegBN modules, one for each modality pair with the reference
            self.reg_bns = nn.ModuleList()
            for _ in range(self.n_modalities - 1):
                reg_bn_instance = RegBN(
                    modalities_channels=[output_dim, output_dim],
                    modalities_dims=[[], []],
                    momentum=regbn_params["momentum"],
                    sigma_THR=regbn_params["sigma_THR"],
                    sigma_MIN=regbn_params["sigma_MIN"],
                    normalize_input=regbn_params["normalize_input"],
                    normalize_output=regbn_params["normalize_output"],
                    affine=regbn_params["affine"],
                    beta1=regbn_params["beta1"],
                    beta2=regbn_params["beta2"]
                )
                self.reg_bns.append(reg_bn_instance)

        self.encoders = encoders
        self.reference_modality_idx = reference_modality_idx if reference_modality_idx != -1 else self.n_modalities - 1
    
    def _norm_regbn(
        self,
        x: list = [
                torch.zeros(128, 249), 
                torch.zeros(128, 249)
        ],
    ) -> tuple:
        n_mods = len(x)
        is_training = self.training
        
        # Select the reference modality based on the hyperparameter
        reference_modality = x[self.reference_modality_idx]
        
        # A list to store the results, initialized to None to preserve order
        results = [None] * n_mods
        
        reg_bn_counter = 0
        last_g_norm = None
        for i in range(n_mods):
            if i == self.reference_modality_idx:
                # This is the reference modality, skip normalizing it against itself.
                continue
            
            processed_modalities = self.reg_bns[reg_bn_counter](
                [x[i], reference_modality],
                is_training=is_training, n_epoch=self.epoch, steps_per_epoch=self.steps_per_epoch
            )
            f_r, g_norm = processed_modalities[0], processed_modalities[1]
            results[i] = f_r
            last_g_norm = g_norm
            reg_bn_counter += 1

        # Place the (potentially output-normalized) reference modality back in its original position.
        results[self.reference_modality_idx] = last_g_norm if last_g_norm is not None else reference_modality
        
        return results
    
    def forward(
            self, 
            x: list = [
                torch.zeros(128, 249), 
                torch.zeros(128, 249)
            ],
            epoch: int = None,
            steps_per_epoch: int = None,
            full_seq: bool = False
    ) -> torch.Tensor:
        n_mod = len(x)
        self.epoch = epoch 
        self.steps_per_epoch = steps_per_epoch
        
        common_latents = []
        total_l1_loss = 0
        for i in range(n_mod):
            encoder_out = self.encoders[i](x[i], full_seq=full_seq)
            # if encoder_out is tuple, take the first element
            if isinstance(encoder_out, tuple):
                encoder_out, l1_loss = encoder_out
                total_l1_loss += l1_loss
            common_latents.append(encoder_out)

        common_latents = self._norm_regbn(common_latents)

        if full_seq:
            return common_latents
        else:
            if total_l1_loss > 0:
                return torch.cat(common_latents, dim=1), total_l1_loss
            else:
                return torch.cat(common_latents, dim=1)


""" HAIM """
class Encoder_HAIM_Vision(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            params: dict = {
                "vit": "vit_b_16",
                "vit_dropout": 0.0,
            }
    ) -> None:
        super().__init__()

        class Full_Sequence_ViT(nn.Module):
            def __init__(
                self,
                encoder,
                dim
            ):
                super().__init__()
                
                self.vit = encoder

                self.linear_proj = nn.Linear(768, dim)
            
            def forward(self, x):
                patch_embeddings = self.vit._process_input(x)
                batch_size = patch_embeddings.shape[0]
                cls_token = self.vit.class_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_token, patch_embeddings], dim=1)
                x = x + self.vit.encoder.pos_embedding
                full_sequence_output = self.vit.encoder(x)
                full_sequence_output = self.linear_proj(full_sequence_output)
                return full_sequence_output

        self.encoder = getattr(models, params["vit"])(weights=None, dropout=params["vit_dropout"], num_classes=10)

        patch_size = int(params["vit"].split('_')[-1])
        self.encoder.conv_proj = nn.Conv2d(1, 768, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.encoder.conv_proj.apply(self._init_weights)

        self.encoder.heads.head = nn.Linear(768, latent_dim)
        self.encoder.heads.head.apply(self._init_weights)

        self.encoder_full_seq = Full_Sequence_ViT(self.encoder, latent_dim)

class Encoder_HAIM_Sequential(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            params: dict = {
                "transformer_num_layers": 4,
                "transformer_num_attention_heads": 2,
                "transformer_dim_feedforward": 256,
                "transformer_dropout": 0.0,
            }
    ) -> None:
        super().__init__()

        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=params["transformer_num_attention_heads"], 
                dim_feedforward=params["transformer_dim_feedforward"], 
                dropout=params["transformer_dropout"], 
                batch_first=True,
            ), num_layers=params["transformer_num_layers"]
        )

        pre_layers = nn.Linear(22, latent_dim)
        pre_layers.apply(self._init_weights)

        self.encoder_full_seq = nn.Sequential(
            pre_layers,
            AddCLSToken(latent_dim),
            AddPE(latent_dim),
            transformer
        )
        self.encoder = nn.Sequential(
            pre_layers,
            AddCLSToken(latent_dim),
            AddPE(latent_dim),
            transformer, 
            ExtractCLSToken()
        )


""" Symile """
class Encoder_Symile_Tabular(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "input_dim": 50,
            "hidden_dims": [256, 512, 256],
            "hidden_dropouts": [0.0, 0.0, 0.0],
        }
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(params["input_dim"], params["hidden_dims"][0]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][0]),
            nn.Dropout(params["hidden_dropouts"][0]),
            nn.Linear(params["hidden_dims"][0], params["hidden_dims"][1]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][1]),
            nn.Dropout(params["hidden_dropouts"][1]),
            nn.Linear(params["hidden_dims"][1], params["hidden_dims"][2]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][2]),
            nn.Dropout(params["hidden_dropouts"][2]),
            nn.Linear(params["hidden_dims"][2], latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            Unsqueeze_Sequence(),
            self.encoder,
        )

        self.apply(self._init_weights)

class Encoder_Symile_Vision(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            params: dict = {
                "vit": "vit_b_16",
                "vit_dropout": 0.0,
            }
    ) -> None:
        super().__init__()

        class Full_Sequence_ViT(nn.Module):
            def __init__(
                self,
                encoder,
                dim
            ):
                super().__init__()
                
                self.vit = encoder

                self.linear_proj = nn.Linear(768, dim)
            
            def forward(self, x):
                patch_embeddings = self.vit._process_input(x)
                batch_size = patch_embeddings.shape[0]
                cls_token = self.vit.class_token.expand(batch_size, -1, -1)
                x = torch.cat([cls_token, patch_embeddings], dim=1)
                x = x + self.vit.encoder.pos_embedding
                full_sequence_output = self.vit.encoder(x)
                full_sequence_output = self.linear_proj(full_sequence_output)
                return full_sequence_output

        self.encoder = getattr(models, params["vit"])(weights=None, dropout=params["vit_dropout"], num_classes=10)
        self.encoder.heads.head = nn.Linear(768, latent_dim)
        self.encoder.heads.head.apply(self._init_weights)

        self.encoder_full_seq = Full_Sequence_ViT(self.encoder, latent_dim)

class Encoder_Symile_Sequential(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            params: dict = {
                "transformer_num_layers": 4,
                "transformer_num_attention_heads": 2,
                "transformer_dim_feedforward": 256,
                "transformer_dropout": 0.0,
            }
    ) -> None:
        super().__init__()

        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=params["transformer_num_attention_heads"], 
                dim_feedforward=params["transformer_dim_feedforward"], 
                dropout=params["transformer_dropout"], 
                batch_first=True,
            ), num_layers=params["transformer_num_layers"]
        )

        pre_layers = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Linear(625, latent_dim)
        )
        pre_layers.apply(self._init_weights)

        self.encoder_full_seq = nn.Sequential(
            pre_layers,
            AddCLSToken(latent_dim),
            AddPE(latent_dim),
            transformer
        )
        self.encoder = nn.Sequential(
            self.encoder_full_seq,
            ExtractCLSToken()
        )


""" INSPECT """
class Encoder_INSPECT_Vision(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "input_dim": 768,
            "hidden_dims": [512, 256, 128],
            "hidden_dropouts": [0.0, 0.0, 0.0],
        }
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(768, params["hidden_dims"][0]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][0]),
            nn.Dropout(params["hidden_dropouts"][0]),
            nn.Linear(params["hidden_dims"][0], params["hidden_dims"][1]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][1]),
            nn.Dropout(params["hidden_dropouts"][1]),
            nn.Linear(params["hidden_dims"][1], params["hidden_dims"][2]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][2]),
            nn.Dropout(params["hidden_dropouts"][2]),
            nn.Linear(params["hidden_dims"][2], latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            self.encoder,
            Unsqueeze_Sequence(),
        )

        self.encoder.apply(self._init_weights)
        self.encoder_full_seq.apply(self._init_weights)

class Encoder_INSPECT_EHR(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "input_dim": 769,
            "hidden_dims": [512, 256, 128],
            "hidden_dropouts": [0.0, 0.0, 0.0],
        }
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(params["input_dim"], params["hidden_dims"][0]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][0]),
            nn.Dropout(params["hidden_dropouts"][0]),
            nn.Linear(params["hidden_dims"][0], params["hidden_dims"][1]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][1]),
            nn.Dropout(params["hidden_dropouts"][1]),
            nn.Linear(params["hidden_dims"][1], params["hidden_dims"][2]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][2]),
            nn.Dropout(params["hidden_dropouts"][2]),
            nn.Linear(params["hidden_dims"][2], latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            self.encoder,
            Unsqueeze_Sequence(),
        )

        self.encoder.apply(self._init_weights)
        self.encoder_full_seq.apply(self._init_weights)


""" MOSI """
class Encoder_MOSI_Language(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        dataset: str = "mosi"
    ) -> None:
        super().__init__()

        if dataset == "mosi":
            # input shapes: l, v, a 
            # torch.Size([32, 50, 768]) torch.Size([32, 50, 20]) torch.Size([32, 50, 5])
            feature_dims = [768, 20, 5]
        elif dataset == "mosei":
            feature_dims = [768, 35, 74]

        self.encoder = nn.Sequential(
            Sequential_Tokenizer_and_Bert(),
            nn.Linear(feature_dims[0], latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            Sequential_Tokenizer_and_Bert(full_seq=True),
            nn.Linear(feature_dims[0], latent_dim),
        )

        self.encoder[1].apply(self._init_weights)
        self.encoder_full_seq[1].apply(self._init_weights)

class Encoder_MOSI_Vision(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        dataset: str = "mosi",
        params: dict = {
            "transformer_num_layers": 6,
            "transformer_num_attention_heads": 1,
            "transformer_dim_feedforward": 1024,
            "transformer_dropout": 0.0,
        }
    ) -> None: 
        super().__init__()

        if dataset == "mosi":
            # input shapes: l, v, a 
            # torch.Size([32, 50, 768]) torch.Size([32, 50, 20]) torch.Size([32, 50, 5])
            feature_dims = [768, 20, 5]
        elif dataset == "mosei":
            feature_dims = [768, 35, 74]

        vit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=params["transformer_num_attention_heads"], 
                dim_feedforward=params["transformer_dim_feedforward"], 
                dropout=params["transformer_dropout"],
                batch_first=True
            ), num_layers=params["transformer_num_layers"]
        )
        self.encoder = nn.Sequential(
            nn.Linear(feature_dims[1], latent_dim),
            AddCLSToken(latent_dim),
            vit,
            ExtractCLSToken(),
        )

        self.encoder_full_seq = nn.Sequential(
            nn.Linear(feature_dims[1], latent_dim),
            vit
        )

        self.encoder[0].apply(self._init_weights)
        self.encoder_full_seq[0].apply(self._init_weights)

class Encoder_MOSI_Audio(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        dataset: str = "mosi",
        params: dict = {
            "transformer_num_layers": 6,
            "transformer_num_attention_heads": 1,
            "transformer_dim_feedforward": 1024,
            "transformer_dropout": 0.0,
        }
    ) -> None: 
        super().__init__()

        if dataset == "mosi":
            # input shapes: l, v, a 
            # torch.Size([32, 50, 768]) torch.Size([32, 50, 20]) torch.Size([32, 50, 5])
            feature_dims = [768, 20, 5]
        elif dataset == "mosei":
            feature_dims = [768, 35, 74]

        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=params["transformer_num_attention_heads"], 
                dim_feedforward=params["transformer_dim_feedforward"], 
                dropout=params["transformer_dropout"], 
                batch_first=True
            ), num_layers=params["transformer_num_layers"]
        )
        self.encoder = nn.Sequential(
            nn.Linear(feature_dims[2], latent_dim),
            AddCLSToken(latent_dim),
            transformer,
            ExtractCLSToken(),
        )
        self.encoder_full_seq = nn.Sequential(
            nn.Linear(feature_dims[2], latent_dim),
            transformer
        )

        self.encoder[0].apply(self._init_weights)
        self.encoder_full_seq[0].apply(self._init_weights)


""" CH-Sims """
class Encoder_CHS_Language(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128
    ) -> None: 
        super().__init__()

        self.encoder = nn.Sequential(
            Sequential_Tokenizer_and_Bert(chinese=True),
            nn.Linear(768, latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            Sequential_Tokenizer_and_Bert(chinese=True, full_seq=True),
            nn.Linear(768, latent_dim),
        )

        self.encoder[1].apply(self._init_weights)
        self.encoder_full_seq[1].apply(self._init_weights)

class Encoder_CHS_Vision(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "num_hidden_layers": 12,
            "num_attention_heads": 1,
            "intermediate_size": 256,
        }
    ) -> None: 
        super().__init__()

        class Helper_Model(nn.Module):
            def __init__(self, full_seq: bool = False):
                super().__init__()
                self.full_seq = full_seq
                cache_dir = "/path/to/data/tmp_hf_cache"
                config = VivitConfig(
                    num_frames=32,
                    image_size=224,
                    hidden_size=latent_dim,
                    num_hidden_layers=params["num_hidden_layers"],
                    num_attention_heads=params["num_attention_heads"],
                    intermediate_size=params["intermediate_size"],
                    cache_dir=cache_dir
                )
                self.model = VivitModel(
                    config=config,
                )
            def forward(self, x):
                if self.full_seq:
                    out = self.model(x)["last_hidden_state"]
                else:
                    out = self.model(x)["pooler_output"]  # only CLS token  # **x
                return out

        self.encoder = nn.Sequential(
            Helper_Model()
        )
        self.encoder_full_seq = nn.Sequential(
            Helper_Model(full_seq=True),
        )

class Encoder_CHS_Audio(Parent_Encoder):

    def __init__(
            self, 
            latent_dim: int = 128,
            params: dict = {
                "num_hidden_layers": 6,
                "num_attention_heads": 1,
                "intermediate_size": 256,
            }
        ) -> None:
        super().__init__()

        class Helper_Model(nn.Module):
            def __init__(self, full_seq: bool = False):
                super().__init__()
                self.full_seq = full_seq
                cache_dir = "/path/to/data/tmp_hf_cache"
                config = Wav2Vec2Config(
                    hidden_size=latent_dim,
                    num_hidden_layers=params["num_hidden_layers"],
                    num_attention_heads=params["num_attention_heads"],
                    intermediate_size=params["intermediate_size"],
                    cache_dir=cache_dir,
                )
                self.model = Wav2Vec2Model(config=config)

            def forward(self, x):
                out = self.model(x)
                if self.full_seq:
                    out = out.last_hidden_state
                else:
                    out = torch.mean(out.last_hidden_state, dim=1)  # only CLS token
                return out

        self.encoder = nn.Sequential(
            Helper_Model()
        )
        self.encoder_full_seq = nn.Sequential(
            Helper_Model(full_seq=True),
        )


""" CREMA-D """
class Encoder_CREMA_D_Video(Parent_Encoder):
    def __init__(
            self, 
            params: dict = {
                "vit_dim": 256,
                "vit_heads": 2,
                "vit_depth": 4,
                "vit_mlp_dim": 256,
                "vit_dropout": 0.1,
                "latent_dim": 256,
            }
        ) -> None:
        super().__init__()
        
        class Image_Encoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.resnet = models.resnet18(weights=None)  # , norm_layer=PermutedLayerNorm  # , norm_layer=None
                self.resnet.fc = nn.Linear(self.resnet.fc.in_features, params["latent_dim"])
                self.resnet.fc.apply(self._init_weights)

            def _init_weights(
                    self,
                    m
                ) -> None: 
                if isinstance(m, (torch.nn.LayerNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            def forward(self, x):
                return self.resnet(x)

        self.encoder = nn.Sequential(
            Image_Encoder()
        )

        self.encoder_full_seq = nn.Sequential(
            Image_Encoder(),
            Unsqueeze_Sequence()
        )

class Encoder_CREMA_D_Audio(Parent_Encoder):
    def __init__(
            self, 
            latent_dim: int = 128,
            params: dict = {
                "vit_dim": 256,
                "vit_heads": 2,
                "vit_depth": 4,
                "vit_mlp_dim": 256,
                "vit_dropout": 0.1,
                "latent_dim": 256,
            }
        ) -> None:
        super().__init__()

        class ResNetForSpectrogram(nn.Module):
            def __init__(self):
                super().__init__()
                self.resnet = models.resnet18(weights=None)  # , norm_layer=None , norm_layer=nn.Identity
                original_conv1 = self.resnet.conv1
                self.resnet.conv1 = nn.Conv2d(
                    in_channels=1,
                    out_channels=original_conv1.out_channels,
                    kernel_size=original_conv1.kernel_size,
                    stride=original_conv1.stride,
                    padding=original_conv1.padding,
                    bias=original_conv1.bias
                )
                num_ftrs = self.resnet.fc.in_features
                self.resnet.fc = nn.Linear(num_ftrs, params["latent_dim"])
                self.resnet.conv1.apply(self._init_weights)
                self.resnet.fc.apply(self._init_weights)
            
            def _init_weights(
                    self,
                    m
                ) -> None: 
                if isinstance(m, (torch.nn.LayerNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.ones_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            def forward(self, x):
                return self.resnet(x)

        self.encoder = nn.Sequential(
            ResNetForSpectrogram()
        )

        self.encoder_full_seq = nn.Sequential(
            ResNetForSpectrogram(),
            Unsqueeze_Sequence()
        )

class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x


""" Mystery MML """
class Encoder_MysteryMML_Tabular(Parent_Encoder):
    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "hidden_dims": [256, 512, 256],
            "hidden_dropouts": [0.0, 0.0, 0.0],
        }
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(50, params["hidden_dims"][0]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][0]),
            nn.Dropout(params["hidden_dropouts"][0]),
            nn.Linear(params["hidden_dims"][0], params["hidden_dims"][1]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][1]),
            nn.Dropout(params["hidden_dropouts"][1]),
            nn.Linear(params["hidden_dims"][1], params["hidden_dims"][2]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][2]),
            nn.Dropout(params["hidden_dropouts"][2]),
            nn.Linear(params["hidden_dims"][2], latent_dim),
        )
        self.encoder.apply(self._init_weights)
        
        self.encoder_full_seq = nn.Sequential(
            self.encoder,
            Unsqueeze_Sequence(),
        )


""" UKB """
class Encoder_UKB_Tabular(Parent_Encoder):
    def __init__(
        self, 
        latent_dim: int = 128,
        params: dict = {
            "input_dim": 769,
            "hidden_dims": [512, 256, 128],
            "hidden_dropouts": [0.0, 0.0, 0.0],
        }
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(params["input_dim"], params["hidden_dims"][0]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][0]),
            nn.Dropout(params["hidden_dropouts"][0]),
            nn.Linear(params["hidden_dims"][0], params["hidden_dims"][1]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][1]),
            nn.Dropout(params["hidden_dropouts"][1]),
            nn.Linear(params["hidden_dims"][1], params["hidden_dims"][2]),
            nn.ReLU(),
            nn.LayerNorm(params["hidden_dims"][2]),
            nn.Dropout(params["hidden_dropouts"][2]),
            nn.Linear(params["hidden_dims"][2], latent_dim),
        )

        self.encoder_full_seq = nn.Sequential(
            self.encoder,
            Unsqueeze_Sequence(),
        )

        self.encoder.apply(self._init_weights)
        self.encoder_full_seq.apply(self._init_weights)