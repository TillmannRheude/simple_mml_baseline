import math
import copy 
import torch 
import torch.nn as nn 

import torchvision.models as models

from transformers import BertModel, BertConfig

class Sequential_Tokenizer_and_Bert(nn.Module):
    def __init__(self,
                 chinese: bool = False):
        super().__init__()

        self.use_finetune = False
        self.use_pretrain_bert = True

        self.chinese = chinese

        if self.use_pretrain_bert:
            self.language_model = BertModel.from_pretrained('bert-base-uncased') if not chinese else BertModel.from_pretrained('bert-base-chinese')
        else: 
            self.language_model = BertModel(BertConfig())

    def forward(self, text):
        if self.chinese: 
            input_ids, input_mask = text[:,0,:].long(), text[:,1,:].float()
            if self.use_finetune: 
                last_hidden_states = self.language_model(input_ids=input_ids,
                                                        attention_mask=input_mask)[0]
            else:
                with torch.no_grad():
                    # freeze language model weights
                    for param in self.language_model.parameters():
                        param.requires_grad = False
                    self.language_model.eval()
                    last_hidden_states = self.language_model(input_ids=input_ids,
                                                            attention_mask=input_mask)[0]
        if not self.chinese:
            input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
            if self.use_finetune:
                last_hidden_states = self.language_model(input_ids=input_ids,
                                                        attention_mask=input_mask,
                                                        token_type_ids=segment_ids)[0]
            else:
                with torch.no_grad():
                    # freeze language model weights
                    for param in self.language_model.parameters():
                        param.requires_grad = False
                    self.language_model.eval()
                    last_hidden_states = self.language_model(input_ids=input_ids,
                                                    attention_mask=input_mask,
                                                    token_type_ids=segment_ids)[0]
        return last_hidden_states[:, 0, :]

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
    def __init__(self, embdim):
        super().__init__()
        self.pe = PositionalEncoding(d_model=embdim)
        
    def forward(self, x):
        return self.pe(x.permute(1, 0, 2)).permute(1, 0, 2)

class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Parent_Encoder(nn.Module):
    
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = None

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

    def forward(
            self, 
            x: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(x)[:, None, :]

class Encoder_Tabular(Parent_Encoder):

    def __init__(
        self, 
        input_dim: int = 249, 
        latent_dim: int = 128,
        hidden_dims: list = [256, 512, 256],
        hidden_dropouts: list = [0.0, 0.0, 0.0],
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(hidden_dropouts[0]),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[1]),
            nn.Dropout(hidden_dropouts[1]),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[2]),
            nn.Dropout(hidden_dropouts[2]),
            nn.Linear(hidden_dims[2], latent_dim),
        )

        self.apply(self._init_weights)

class Encoder_Image(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            vit_dropout: float = 0.0,
            grayscale: bool = False
    ) -> None:
        super().__init__()

        self.encoder = models.vit_b_16(weights=None, dropout=vit_dropout, num_classes=10)

        if grayscale: 
            self.encoder.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
            self.encoder.conv_proj.apply(self._init_weights)

        self.encoder.heads.head = nn.Linear(768, latent_dim)
        self.encoder.heads.head.apply(self._init_weights)

class Encoder_Sequence(Parent_Encoder):
    
    def __init__(
            self,
            latent_dim: int = 128,
            use_pre_convs: bool = True, 
            use_pre_linear: bool = False, 
            vit_nhead: int = 2,
            vit_dropout: float = 0.0,
            vit_dim_feedforward: int = 256,
            vit_num_layers: int = 4
    ) -> None:
        super().__init__()

        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=vit_nhead, 
                dim_feedforward=vit_dim_feedforward, 
                dropout=vit_dropout, 
                batch_first=True,
            ), num_layers=vit_num_layers
        )

        if use_pre_convs: 
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
        elif use_pre_linear: 
            pre_layers = nn.Linear(22, latent_dim)
            pre_layers.apply(self._init_weights)
        else: 
            pre_layers = nn.Identity()

        self.encoder = nn.Sequential(
            pre_layers,
            AddCLSToken(latent_dim),
            AddPE(latent_dim),
            transformer, 
            ExtractCLSToken()
        )

class Encoder_MNIST(Parent_Encoder):

    def __init__(
        self, 
        latent_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            Reshape(), 
            nn.Linear(in_features=14*28, out_features=512),
            nn.ReLU(), 
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(), 
            nn.Linear(1024, 512), 
            nn.ReLU(), 
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, latent_dim),  
        )

        self.apply(self._init_weights)

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

        self.encoder[1].apply(self._init_weights)

class Encoder_MOSI_Vision(Parent_Encoder):

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

        vit = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=4, 
                dim_feedforward=512, 
                dropout=0.0,
                batch_first=True
            ), num_layers=14
        )
        self.encoder = nn.Sequential(
            nn.Linear(feature_dims[1], latent_dim),
            AddCLSToken(latent_dim),
            vit,
            ExtractCLSToken(),
        )

        self.encoder[0].apply(self._init_weights)

class Encoder_MOSI_Audio(Parent_Encoder):

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

        transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=latent_dim, 
                nhead=4, 
                dim_feedforward=512, 
                dropout=0.0, 
                batch_first=True
            ), num_layers=4
        )
        self.encoder = nn.Sequential(
            nn.Linear(feature_dims[2], latent_dim),
            AddCLSToken(latent_dim),
            transformer,
            ExtractCLSToken(),
        )

        self.encoder[0].apply(self._init_weights)


class Encoders(nn.Module):

    def __init__(
            self, 
            encoders: nn.ModuleList = nn.ModuleList([Encoder_Tabular(), Encoder_Tabular()])
    ) -> None: 
        super().__init__()

        self.encoders = encoders
    
    def forward(
            self, 
            x: list = [
                torch.zeros(128, 249), 
                torch.zeros(128, 249)
            ]
    ) -> torch.Tensor:
        
        common_latents = []
        for i in range(len(x)):
            common_latents.append(self.encoders[i](x[i]))
        
        return torch.cat(common_latents, dim=1)