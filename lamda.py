from peft import PeftModel, get_peft_model, TaskType, LoraConfig
from peft.tuners.lora.layer import LoraLayer
from typing import Any, List, Optional, Union
from types import MethodType
import torch
import torch.nn as nn
def forward_adapter(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                sigma = self.sigma[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(sigma(lora_A(dropout(x)))) * scaling

        result = result.to(previous_dtype)
        return result


def get_lamda_model(model):

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            rank = module.lora_A['default'].out_features
            module.sigma = nn.ModuleDict({})
            module.sigma['default'] = nn.Linear(rank, rank, bias = False)
            module.sigma['default'].weight = nn.Parameter(module.sigma['default'].weight.bfloat16())
            module.forward = MethodType(forward_adapter, module)

    for pname, p in model.named_parameters():
        if ('lora_A' in pname):
            p.requires_grad = False
        if ('lora_B' in pname and ('bias' in pname)):
            p.requires_grad = False



