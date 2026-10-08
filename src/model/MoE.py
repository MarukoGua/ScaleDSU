import torch
from torch import nn, einsum
import torch.nn.functional as F

import einx
from einops import rearrange, reduce


def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale


class GEGLU(nn.Module):
    def __init__(
        self,
        dim,
        mult_bias = True
    ):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x * self.mult_bias


class Expert(nn.Module):
    def __init__(self, in_dim, hidden_mult=4, multi_bias=True, prenorm=False):
        super(Expert, self).__init__()
        # dim_hidden = int(in_dim*hidden_mult*2/3)
        dim_hidden = int(in_dim/2)
        self.net = nn.Sequential(
            RMSNorm(in_dim) if prenorm else nn.Identity(),
            nn.Linear(in_dim, hidden_mult),
            nn.GELU(),
            nn.Dropout(p=0.3)
            # GEGLU(dim_hidden, mult_bias=multi_bias),
            # nn.Linear(dim_hidden, in_dim)
        )

    #     self.apply(self.init_)

    # def init_(self, module):
    #     if isinstance(module, nn.Linear):
    #         dim = module.weight.shape[0]
    #         std = dim ** -0.5

    #         module.weight.data.uniform_(-std, std)
    #         module.bias.data.uniform_(-std, std)
    
    def forward(self, x):
        out = self.net(x)
        return out


class Experts(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        shape = x.shape

        x = rearrange(x, 'b e n d -> e b n d')
        outputs = []
        for expert, expert_input in zip(self.experts, x):
            out = expert(expert_input)
            outputs.append(out)
        
        if len(outputs) > 0:
            outputs = torch.stack(outputs)
        else:
            outputs = torch.empty_like(x, requires_grad=self.training)
        outputs = rearrange(outputs, 'e b n d -> b e n d')
        # assert outputs.shape == shape
        return outputs


class TopNGating(nn.Module):
    def __init__(
            self, 
            dim, 
            num_gates, 
            eps=1e-9, 
            top_n=2, 
            threshold_train=0.2, 
            threshold_eval=0.2, 
            capacity_factor_train=1.25, 
            capacity_factor_eval=2., 
            straight_through_dispatch_tensor=True
        ):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias = False)
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = ((threshold_train,)*top_n_minus_1)
        threshold_eval = ((threshold_eval,)*top_n_minus_1)
        
        self.register_buffer('threshold_train', torch.tensor([eps, *threshold_train]))
        self.register_buffer('threshold_eval', torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval        

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer('zero', torch.zeros((1,)), persistent = False)

    def forward(self, x, noise_gates=False, noise_mult=1.):
        *_, b, group_size, dim, dtype, top_n, num_gates, eps = *x.shape, x.dtype, self.top_n, self.num_gates, self.eps
        suffix = 'train' if self.training else 'eval'

        threshold = getattr(self, f'threshold_{suffix}')
        capacity_factor = getattr(self, f'capacity_factor_{suffix}')

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, 4)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x) # (b, c, num_gates)

        maybe_noised_gate_logits = gate_logits

        if noise_gates:
            noise = torch.zeros_like(maybe_noised_gate_logits).uniform_(0, 1)
            log = lambda x: torch.log(x.clamp(min=1e-20))
            noise = -log(-log(noise))

            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim=-1) # (b, c, num_gates)

        # find top N experts per position
        gates, gate_indices = torch.topk(raw_gates, top_n, dim=-1) # (b, c, top_n)

        # move the top_n dimension to be the first
        gates = rearrange(gates, '... k -> k ...')  # (top_n, b, c)
        gate_indices = rearrange(gate_indices, '... k -> k ...') # (top_n, b, c)

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates) # (top_n, b, c, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0]

        # normalize top-n gate scores

        denom = reduce(gates, 'k ... -> 1 ...', 'sum').clamp(min=eps) # (1, b, c) sum along first top-n indices, sum top-n probabilities up?
        gates = gates / denom
        
        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts
        probs = torch.zeros_like(gates).uniform_(0., 1.)

        should_route = probs < einx.divide('k b n, k -> k b n', gates, threshold.clamp(min = eps))

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), '... -> ... 1')
        pre_padding = (0, 0) * 1
        mask_cumsum = F.pad(mask, (*pre_padding, 1, -1)).cumsum(dim=-2) # cumsum along channel dimension (top_n, b, c, num_gates)

        positions = []
        prev_expert_count = 0.

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]
            
            # Remove the elements that don't fit. exceed expert capacity!
            mask[n] *= (position_in_expert < expert_capacity_f).float() # (b, c, num_gates)

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = reduce(mask[n], '... n e -> ... 1 e', 'sum') + prev_expert_count

            position_in_expert = reduce(position_in_expert, '... n e -> ... n', 'sum') # (b, c)
            positions.append(position_in_expert)

        positions = torch.stack(positions) # (top_n, b, c)

        
        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, '... n e -> ... n', 'sum')

        # (k, batch, sequence) - weighted assignment
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            'k b n, k b n, k b n e, k b n c -> k b n e c',
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity)
        )

        combine_tensor = reduce(combine_tensor, 'k b n e c -> b n e c', 'sum')

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, '... n e -> ... e', 'mean')
            density_1_proxy = reduce(raw_gates, '... n e -> ... e', 'mean') # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim = -1)
            router_z_loss = torch.square(router_z_loss)            
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return dispatch_tensor, combine_tensor, balance_loss, router_z_loss



class MoE(nn.Module):
    def __init__(
        self, 
        dim,
        num_experts = 16,
        expert_hidden_mult = 4,
        threshold_train = 0.2,
        threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        gating_top_n = 2,
        balance_loss_coef = 1e-2,
        router_z_loss_coef = 1e-3,
        allow_var_seq_len = False
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        self.gate = TopNGating(
            dim,
            top_n = gating_top_n,
            num_gates = num_experts,
            threshold_train = threshold_train,
            threshold_eval = threshold_eval,
            capacity_factor_train = capacity_factor_train,
            capacity_factor_eval = capacity_factor_eval
        )

        experts = [Expert(in_dim=dim, hidden_mult=expert_hidden_mult) for _ in range(num_experts)]
        self.experts = Experts(experts)

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef


    def forward(self, x, noise_gates=False, noise_mult=1.):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates, noise_mult)

        # dispatch
        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.
        expert_outputs = self.experts(expert_inputs)

        # combine
        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return output, total_aux_loss, balance_loss, router_z_loss


class SparseMoEBlock(nn.Module):
    def __init__(self, moe, add_ff_before=False, add_ff_after=True):
        super().__init__()
        dim = moe.dim
        self.moe = moe
        self.moe_prenorm = RMSNorm(dim)

        self.ff_before = Expert(dim, prenorm=True) if add_ff_before else None
        self.ff_after = Expert(dim, prenorm=True) if add_ff_after else None

    def forward(self, x, noise_gates=False, noise_mult=1.):
        if self.ff_before is not None:
            x = self.ff_before(x) + x
        
        residual = x

        moe_out, total_aux_loss, balance_loss, router_z_loss = self.moe(self.moe_prenorm(x), noise_gates = noise_gates, noise_mult = noise_mult)

        x = moe_out + residual

        # feedforward after

        if self.ff_after is not None:
            x = self.ff_after(x) + x

        return x, total_aux_loss, balance_loss, router_z_loss

    

if __name__ == '__main__':
    # hyperparams
    moe = MoE(dim=3, num_experts=4)
    inputs = torch.randn(1,20,3)
    out, total_aux_loss, balance_loss, router_z_loss = moe(inputs)
    print(out.shape)
    print(total_aux_loss)
    print(balance_loss, router_z_loss)

    # moe_block = SparseMoEBlock(moe, True, True)
    # out, total_aux_loss, balance_loss, router_z_loss = moe_block(inputs)
    # print(out.shape)
    # print(total_aux_loss)
    # print(balance_loss, router_z_loss)