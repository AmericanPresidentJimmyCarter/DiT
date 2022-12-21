import os

from copy import deepcopy

import torch

from torch import nn


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None, ema_model_path=None):
        super(ModelEma, self).__init__()
        if ema_model_path is not None and os.path.exists(ema_model_path):
            resume = True
        else:
            resume = False

        if resume:
            self.module = torch.load(ema_model_path)
        else:
            # Make a copy of the model for accumulating moving average of
            # weights.
            self.module = deepcopy(model)
            self.module.eval()
        self.decay = decay

        # Perform ema on different device from model if set.
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
