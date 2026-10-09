import copy

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .conv import EncoderBlock, ACT2FN, DenseLayers
from .dsu import DistributionUncertainty
from .MoE import MoE

class GaussianNoiseLayer(nn.Module):
    def __init__(self, noise_level=0.1, **kwargs):
        """
        Gaussian noise injection
        :param noise_level: level of noise injected。
        :param adaptive: 是否自适应输入特征的幅度。
        """
        super(GaussianNoiseLayer, self).__init__()
        self.noise_level = noise_level

    def forward(self, x, *args, **kwargs):
        # nn.Module 自带 self.training 属性。
        # 当模型执行 model.eval() 时，此处的 self.training 会自动变为 False，从而安全地跳过噪声注入
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x)
            return (1-self.noise_level) * x + self.noise_level * noise
        return x


class DropoutLayer(nn.Module):
    def __init__(self, dropout=0.1, **kwargs):
        super(DropoutLayer, self).__init__()
        self.dropout=dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        print(f'add dropout! {dropout}')
    
    def forward(self, x, *args, **kwargs):
        if self.training and self.dropout > 0:
            return self.dropout_layer(x)
        return x



class AblationDSU(nn.Module):
    def __init__(self, config, perturbation, level, denser, uncertainty=1.0, factor=5.0, dropout=0.0):
        super().__init__() 

        self.embedder = nn.Sequential(
            nn.Conv1d(1, config.start_channels, config.patch_size, stride=1, padding=(config.patch_size-1)//2),
            nn.BatchNorm1d(config.start_channels),
            ACT2FN[config.hidden_act]()
            )
        
        if perturbation == 'gaussian':
            perturb_func = GaussianNoiseLayer(float(level))
        elif perturbation == 'dsu':
            perturb_func = DistributionUncertainty(uncertainty=bool(uncertainty), factor=int(factor))
        elif perturbation == 'dropout':
            perturb_func = DropoutLayer(dropout=float(dropout))
        
        self.perturb1 = perturb_func
        self.pool1 = nn.AvgPool1d(config.embedding_pool, config.embedding_stride)

        self.encoder = nn.Sequential()

        inplanes = config.start_channels
        k = config.encoder_kernel
        pk = config.encoder_pool_kernel
        ps = config.encoder_pool_stride
        for c in config.encoder_channels:
            self.encoder.append(EncoderBlock(inplanes, c, k, config.hidden_act))
            self.encoder.append(perturb_func)
            self.encoder.append(nn.AvgPool1d(pk, ps))
            inplanes = c

        self.denser_type = denser
        if denser == 'moe':
            expert_hidden_mult = config.kwargs['expert_hidden_mult']
            num_experts = config.kwargs['num_experts']
            moe = MoE(int(config.dense_in_dim/inplanes), num_experts=num_experts, expert_hidden_mult=expert_hidden_mult)
            self.denser = moe
            self.classifier = nn.Linear(inplanes*expert_hidden_mult, config.num_labels)
        elif denser == 'mlp':
            self.denser = DenseLayers(config)
            if config.mlp_depths == 0:
                out_dim = config.dense_in_dim
            else:
                out_dim = config.mlp_widths
            self.classifier = nn.Linear(out_dim, config.num_labels)
        
        self.apply(self._init_weights)

        self.ce_loss = nn.CrossEntropyLoss()
    
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
    
    def path(self, x, add_uncertainty):
        x = x.unsqueeze(1).float()
        x = self.embedder(x)
        x = self.pool1(self.perturb1(x, False))
        for enc in self.encoder:
            if isinstance(enc, DistributionUncertainty):
                x = enc(x, add_uncertainty)
            else:
                x = enc(x)
        return x

    def dsu_scale(self, x, add_uncertainty):

        if add_uncertainty:
            clean = copy.deepcopy(x)
            clean = self.path(clean, add_uncertainty=False)
            # bs = feat.shape[0]

            noisy = self.path(x, add_uncertainty)
            if self.denser_type == 'mlp':
                clean = clean.flatten(1)
                clean = self.denser(clean)
                noisy = noisy.flatten(1)
                noisy = self.denser(noisy)
                aux_loss = 0.0
            elif self.denser_type == 'moe':
                clean = self.denser(clean)[0]
                clean = clean.flatten(1)
                noisy, aux_loss, _, _ = self.denser(noisy)
                noisy = noisy.flatten(1)
            clean = self.classifier(clean)
            noisy = self.classifier(noisy)

            return clean, noisy, aux_loss
        else:
            feat = self.path(x, add_uncertainty=False)
            if self.denser_type == 'mlp':
                feat = feat.flatten(1)
                feat = self.denser(feat)
            elif self.denser_type == 'moe':
                feat = self.denser(feat)[0]
                feat = feat.flatten(1)
            feat = self.classifier(feat)
            return feat


    def forward(self, x, label=None, add_uncertainty=False):

        aux_loss = 0.0
        ce_loss = 0.0
        if add_uncertainty:
            clean, noisy, aux_loss = self.dsu_scale(x, add_uncertainty)
            clean_ce_loss = self.ce_loss(clean, label)
            noisy_ce_loss = self.ce_loss(noisy, label)
            ce_loss = (clean_ce_loss + noisy_ce_loss) / 2
        else:
            clean = self.dsu_scale(x, add_uncertainty)
            ce_loss = self.ce_loss(clean, label)
        return clean, aux_loss, ce_loss



class AblationInput(nn.Module):
    def __init__(self, config, perturbation, level, denser, uncertainty=1.0, factor=5.0, dropout=0.0):
        super().__init__() 
    
        if perturbation == 'gaussian':
            perturb_func = GaussianNoiseLayer(float(level))
            print(float(level))
        elif perturbation == 'dsu':
            perturb_func = DistributionUncertainty(uncertainty=bool(uncertainty), factor=int(factor))
            print(bool(uncertainty), int(factor))
        elif perturbation == 'dropout':
            perturb_func = DropoutLayer(dropout=float(dropout))

        self.perturb = perturb_func
        self.embedder = nn.Sequential(
            nn.Conv1d(1, config.start_channels, config.patch_size, stride=1, padding=(config.patch_size-1)//2),
            nn.BatchNorm1d(config.start_channels),
            ACT2FN[config.hidden_act]()
            )
        
        self.pool1 = nn.AvgPool1d(config.embedding_pool, config.embedding_stride)

        self.encoder = nn.Sequential()

        inplanes = config.start_channels
        k = config.encoder_kernel
        pk = config.encoder_pool_kernel
        ps = config.encoder_pool_stride
        for c in config.encoder_channels:
            self.encoder.append(EncoderBlock(inplanes, c, k, config.hidden_act))
            self.encoder.append(nn.AvgPool1d(pk, ps))
            inplanes = c

        self.denser_type = denser
        if denser == 'moe':
            expert_hidden_mult = config.kwargs['expert_hidden_mult']
            num_experts = config.kwargs['num_experts']
            moe = MoE(int(config.dense_in_dim/inplanes), num_experts=num_experts, expert_hidden_mult=expert_hidden_mult)
            self.denser = moe
            self.classifier = nn.Linear(inplanes*expert_hidden_mult, config.num_labels)
        elif denser == 'mlp':
            self.denser = DenseLayers(config)
            if config.mlp_depths == 0:
                out_dim = config.dense_in_dim
            else:
                out_dim = config.mlp_widths
            self.classifier = nn.Linear(out_dim, config.num_labels)
        
        self.apply(self._init_weights)

        self.ce_loss = nn.CrossEntropyLoss()
    
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
    
    def path(self, x, add_uncertainty):
        x = x.unsqueeze(1).float()
        x = self.perturb(x, add_uncertainty)
        x = self.embedder(x)
        x = self.pool1(x)
        x = self.encoder(x)
        return x

    def dsu_scale(self, x, add_uncertainty):

        if add_uncertainty:
            clean = copy.deepcopy(x)
            clean = self.path(clean, add_uncertainty=False)
            noisy = self.path(x, add_uncertainty)
            if self.denser_type == 'mlp':
                clean = clean.flatten(1)
                clean = self.denser(clean)
                noisy = noisy.flatten(1)
                noisy = self.denser(noisy)
                aux_loss = 0.0
            elif self.denser_type == 'moe':
                clean = self.denser(clean)[0]
                clean = clean.flatten(1)
                noisy, aux_loss, _, _ = self.denser(noisy)
                noisy = noisy.flatten(1)
            clean = self.classifier(clean)
            noisy = self.classifier(noisy)

            return clean, noisy, aux_loss
        else:
            feat = self.path(x, add_uncertainty=False)
            if self.denser_type == 'mlp':
                feat = feat.flatten(1)
                feat = self.denser(feat)
            elif self.denser_type == 'moe':
                feat = self.denser(feat)[0]
                feat = feat.flatten(1)
            feat = self.classifier(feat)
            return feat


    def forward(self, x, label=None, add_uncertainty=False):

        aux_loss = 0.0
        ce_loss = 0.0

        if add_uncertainty:
            clean, noisy, aux_loss = self.dsu_scale(x, add_uncertainty)
            clean_ce_loss = self.ce_loss(clean, label)
            noisy_ce_loss = self.ce_loss(noisy, label)
            ce_loss = (clean_ce_loss + noisy_ce_loss) / 2
        else:
            clean = self.dsu_scale(x, add_uncertainty)
            ce_loss = self.ce_loss(clean, label)
        return clean, aux_loss, ce_loss