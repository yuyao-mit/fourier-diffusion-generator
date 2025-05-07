
# calculate_flop.py

from thop import profile, clever_format
import torch
from typing import Tuple

def calculate_flop(model, inputs: Tuple[torch.Tensor, ...]) -> Tuple[str, str]:
    
    assert isinstance(inputs, tuple), "inputs must be a tuple, e.g., (x,) even for a single input"
    # assert all(isinstance(x, torch.Tensor) for x in inputs), "all elements in inputs must be torch.Tensor"

    model.eval()

    flops, params = profile(model, inputs=inputs)

    print(f"Raw FLOPs: {flops:.4e}, Raw Params: {params:.4e}")
    flops_str, params_str = clever_format([flops, params], "%.3f")

    print(f"Formatted FLOPs: {flops_str}, Formatted Params: {params_str}")

    return flops_str, params_str
