from torch import nn, log, exp, mean, Tensor

from .conv import ConvNet
from .unet import UNetConfig, UNet


class SCAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        cnet = ConvNet(config)
        gconfig = UNetConfig()
        unet = UNet(gconfig)

        self.cnet = cnet
        self.embedder = cnet.embedder
        self.encoder = cnet.encoder
        self.denser = cnet.denser
        self.unet = unet
        self.classifier = cnet.classifier
        self.std = nn.Parameter(Tensor([1, 1, 1]), requires_grad=True)
    
    def forward(self, traces, plts=None, is_noisy=False):
        hidden_states = self.embedder(traces.unsqueeze(1))
        print(hidden_states.shape)
        input()
        hidden_states = self.encoder(hidden_states)
        hidden_states = hidden_states.flatten(start_dim=1, end_dim=2)
        hidden_states = self.denser(hidden_states)
        prediction = self.classifier(hidden_states)

        if is_noisy:
            denoised_states = self.unet(hidden_states.unsqueeze(1), plts).squeeze(1)
            prediction = self.classifier(denoised_states)
            return denoised_states, prediction
        else:
            return hidden_states, prediction
        
    def criterion(self, losses):
        log_vars = log(self.std ** 2)
        precisions = exp(-log_vars)
        total_loss = 0
        
        for i, loss in enumerate(losses):
            total_loss += precisions[i]*loss + log_vars[i] ** 2
        return mean(total_loss)