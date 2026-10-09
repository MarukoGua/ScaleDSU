import copy

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .conv import EncoderBlock, ACT2FN
from .MoE import MoE



class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, eps=1e-6, uncertainty=1.0, factor=5, **kwargs):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.factor = factor
        self.uncertainty = uncertainty

    def _reparameterize(self, mu, std):
        factor = torch.randint(1, self.factor+1, std.size()).cuda()
        epsilon = torch.randn_like(std) * factor
        return mu + epsilon * std
        

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, add_uncertainty=False):
        if (self.uncertainty == 0) or (add_uncertainty is False) or (not self.training):
            return x
        mean = x.mean(dim=-1, keepdim=False)
        std = (x.var(dim=-1, keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std) 

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1)) / std.reshape(x.shape[0], x.shape[1],  1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1) + beta.reshape(x.shape[0], x.shape[1], 1)
        return x


class ScaleDSU(nn.Module):
    def __init__(self, config, perturbation=DistributionUncertainty, uncertainty=1.0, factor=5.0):
        super().__init__() 

        self.embedder = nn.Sequential(
            nn.Conv1d(1, config.start_channels, config.patch_size, stride=1, padding=(config.patch_size-1)//2),
            nn.BatchNorm1d(config.start_channels),
            ACT2FN[config.hidden_act]()
            )
        self.perturb1 = perturbation(uncertainty=uncertainty, factor=factor)
        self.pool1 = nn.AvgPool1d(config.embedding_pool, config.embedding_stride)

        self.encoder = nn.Sequential()

        inplanes = config.start_channels
        k = config.encoder_kernel
        pk = config.encoder_pool_kernel
        ps = config.encoder_pool_stride
        for c in config.encoder_channels:
            self.encoder.append(EncoderBlock(inplanes, c, k, config.hidden_act))
            self.encoder.append(perturbation(uncertainty=uncertainty, factor=factor))
            self.encoder.append(nn.AvgPool1d(pk, ps))
            inplanes = c

        expert_hidden_mult = config.kwargs['expert_hidden_mult']
        num_experts = config.kwargs['num_experts']
        moe = MoE(int(config.dense_in_dim/inplanes), num_experts=num_experts, expert_hidden_mult=expert_hidden_mult)
        self.denser = moe
        
        self.classifier = nn.Linear(inplanes*expert_hidden_mult, config.num_labels)
        
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
            clean, _, _ , _ = self.denser(clean)
            clean = clean.flatten(1)
            clean = self.classifier(clean)

            noisy = self.path(x, add_uncertainty)
            noisy, aux_loss, _, _ = self.denser(noisy)
            noisy = noisy.flatten(1)
            noisy = self.classifier(noisy)

            return clean, noisy, aux_loss
        else:
            feat = self.path(x, add_uncertainty=False)
            feat, _, _ , _ = self.denser(feat)
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

