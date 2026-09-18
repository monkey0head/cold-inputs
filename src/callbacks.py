"""
Pytorch Lightning callbacks.
"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import grad_norm


class LogGradNormCallback(pl.Callback):
    """
    Callback for periodically logging the norm of the model's gradients.
    """

    def __init__(self, log_every_n_steps: int = 100, norm_type: float = 2.0, log_per_layer: bool = False,
                 include_patterns: list[str] = None, exclude_patterns: list[str] = None):

        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type
        self.log_per_layer = log_per_layer
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer):
        
        if trainer.global_step % self.log_every_n_steps == 0:

            norms = grad_norm(pl_module, norm_type=self.norm_type)
            
            total_norm_key = f"grad_{self.norm_type}_norm_total"
            if total_norm_key in norms:
                pl_module.log("grad_norm/total", norms[total_norm_key], 
                              on_step=True, on_epoch=False,
                              prog_bar=False, logger=True)
            
            if self.log_per_layer:
                for key, value in norms.items():
                    if key != total_norm_key and self._should_log_layer(key):
                        pl_module.log(f"grad_norm/{key}", value, 
                                      on_step=True, on_epoch=False,
                                      prog_bar=False, logger=True)

    def _should_log_layer(self, layer_name: str) -> bool:
        if not self.log_per_layer:
            return False
            
        if self.include_patterns:
            if not any(pattern in layer_name for pattern in self.include_patterns):
                return False
                
        if self.exclude_patterns:
            if any(pattern in layer_name for pattern in self.exclude_patterns):
                return False
                
        return True


class LogLayerNormGainCallback(pl.Callback):
    """Periodically log the L2 norm of each LayerNorm gain (weight) vector.

    Read-only: walks ``nn.LayerNorm`` submodules and logs ``||weight||`` so a growing
    LayerNorm gain (an early signature of the divergence) is visible directly, without
    modifying the backbone. Architecture-agnostic: matches any ``nn.LayerNorm``.
    """

    def __init__(self, log_every_n_steps: int = 30, norm_type: float = 2.0,
                 log_per_layer: bool = True):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.norm_type = norm_type
        self.log_per_layer = log_per_layer

    @staticmethod
    def _clean(name: str) -> str:
        for prefix in ("model._orig_mod.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name.replace("transformer_model.", "")

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer):

        if trainer.global_step % self.log_every_n_steps != 0:
            return

        total = 0.0
        max_norm = 0.0
        found = False
        with torch.no_grad():
            for name, module in pl_module.named_modules():
                if not isinstance(module, torch.nn.LayerNorm) or module.weight is None:
                    continue
                found = True
                w_norm = module.weight.detach().norm(self.norm_type).item()
                total += w_norm ** self.norm_type
                max_norm = max(max_norm, w_norm)
                if self.log_per_layer:
                    pl_module.log(f"ln_gain_norm/{self._clean(name)}", w_norm,
                                  on_step=True, on_epoch=False, prog_bar=False, logger=True)

        if found:
            pl_module.log("ln_gain_norm/total", float(total ** (1.0 / self.norm_type)),
                          on_step=True, on_epoch=False, prog_bar=False, logger=True)
            pl_module.log("ln_gain_norm/max", max_norm,
                          on_step=True, on_epoch=False, prog_bar=False, logger=True)
