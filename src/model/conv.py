import json
from math import floor

from torch import nn
from timm.layers import trunc_normal_

ACT2FN = {
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "leaky relu": nn.LeakyReLU,
    "swish": nn.SiLU
}

class ConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Conv1d(1, config.start_channels, config.patch_size, 1, (config.patch_size-1)//2),
            nn.BatchNorm1d(config.start_channels),
            ACT2FN[config.hidden_act]()
        )
        self.pooler = nn.AvgPool1d(
            kernel_size=config.embedding_pool, 
            stride=config.embedding_stride
        )
    
    def forward(self, x):
        embedding = self.embedder(x)
        embedding = self.pooler(embedding)
        return embedding


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, actv):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel, 1, padding=(kernel-1)//2),
            nn.BatchNorm1d(out_channel),
            ACT2FN[actv](),
            nn.Conv1d(out_channel, out_channel, kernel, 1, padding=(kernel-1)//2)
        )
        self.res_conv = nn.Conv1d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, x):
        h = self.block(x)
        return h + self.res_conv(x)


class ConvEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        inplanes = config.start_channels
        k = config.encoder_kernel
        pk = config.encoder_pool_kernel
        ps = config.encoder_pool_stride
        self.stages = nn.Sequential()
        
        for c in config.encoder_channels:
            self.stages.append(EncoderBlock(inplanes, c, k, config.hidden_act))
            self.stages.append(nn.AvgPool1d(pk, ps))
            inplanes = c


    def forward(self, hidden_state):
        if not self.stages:
            return hidden_state

        for stage_module in self.stages:
            hidden_state = stage_module(hidden_state)
        
        return hidden_state

class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, drop=0.):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.activation = ACT2FN[activation]()
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        out = self.drop(out)
        return out


class DenseLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        actv = config.hidden_act
        drop = config.dropout
        in_dim = config.dense_in_dim
        self.stages = nn.Sequential()

        for _ in range(config.mlp_depths):
            self.stages.append(DenseBlock(in_dim, config.mlp_widths, actv, drop))
            in_dim = config.mlp_widths

    def forward(self, hidden_state):
        for stage_module in self.stages:
            hidden_state = stage_module(hidden_state)
        return hidden_state

class ConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedder = ConvEmbedding(config)
        self.encoder = ConvEncoder(config)
        self.denser = DenseLayers(config)

        if config.mlp_depths == 0:
            out_dim = config.dense_in_dim
        else:
            out_dim = config.mlp_widths
        self.classifier = nn.Linear(out_dim, config.num_labels)
    
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, input_sequences, return_hidden_states = False):         
        input_sequences = input_sequences.unsqueeze(1).float()
        embedding_output = self.embedder(input_sequences)
        encoder_output = self.encoder(embedding_output)
        encoder_output = encoder_output.flatten(start_dim=1, end_dim=2)
        # encoder_output = mean(encoder_output, dim=-1)
        dense_output = self.denser(encoder_output)
        class_output = self.classifier(dense_output)

        if return_hidden_states:
            hidden_states = (embedding_output, encoder_output, dense_output)
            return class_output, hidden_states

        return class_output

        return dim * self.encoder_channels[-1]

class ConvConfig:
    def __init__(
            self,
            start_channels: int=1,
            patch_size: int=7,
            embedding_pool: int=2,
            embedding_stride: int = 2, 
            encoder_depths: int = 0,
            encoder_kernel: int = 3,
            encoder_pool_kernel: int = 2,
            encoder_pool_stride: int = 2,
            mlp_depths: int = 2,
            mlp_widths: int = 32,
            hidden_act: str ='selu',
            num_labels: int = 256, 
            dropout: float = 0.2,
            **kwargs
    ):
        
        super().__init__()
        self.start_channels = start_channels
        self.patch_size = patch_size
        self.embedding_pool = embedding_pool
        self.embedding_stride = embedding_stride

        self.encoder_depths = encoder_depths
        self.encoder_kernel = encoder_kernel
        self.encoder_pool_kernel = encoder_pool_kernel
        self.encoder_pool_stride = encoder_pool_stride

        self.mlp_depths = mlp_depths
        self.mlp_widths = mlp_widths
       
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.dropout = dropout

        self.sample_length = kwargs['length']
        self.kwargs = kwargs

    @classmethod
    def from_config_file(cls, config_file_path, **kwargs):
        try:
            with open(config_file_path, "r", encoding="utf-8") as reader:
                text = reader.read()
            config = json.loads(text)
        except:
            raise ValueError(f'Check your config path: {config_file_path} is a valid json file.')
        config = cls(**config, **kwargs)
        
        for key, val in kwargs.items():
            if hasattr(config, key):
                print(config, key)
                setattr(config, key, val)
        config.compute_dim()

        return config


    def compute_dim(self):
        self.encoder_channels = [self.start_channels*(2**((i+2)//2)) for i in range(1, self.encoder_depths+1)]
        self.embedding_output_dim = self.compute_embedding_output_dim()
        self.dense_in_dim = self.compute_encoder_output_dim()

    def compute_embedding_output_dim(self):
        kernel = self.patch_size
        padding = (kernel - 1) // 2
        stride = 1
        conv_dim = floor( ( self.sample_length + 2*padding - (kernel-1) -1 ) / stride + 1 )
        out_dim = floor((conv_dim - self.embedding_pool) / self.embedding_stride + 1)
        return out_dim
    
    def compute_encoder_output_dim(self):
        if self.encoder_depths == 0:
            return self.embedding_output_dim * self.start_channels
        dim = self.embedding_output_dim
        pk = self.encoder_pool_kernel
        ps = self.encoder_pool_stride
        for _ in self.encoder_channels:
            p = (self.encoder_kernel-1)//2
            dim = floor((dim + 2*p - (self.encoder_kernel-1) -1 ) + 1)
            dim = floor((dim-pk) / ps + 1)
        return dim * self.encoder_channels[-1]