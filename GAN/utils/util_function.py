import torch
from torch import Tensor
from typing import Tuple,Optional
import numpy as np

def update_average(model_tgt, model_src, beta):
    """
    function to calculate the Exponential moving averages for the Generator weights
    This function updates the exponential average weights based on the current training
    Args:
        model_tgt: target model
        model_src: source model
        beta: value of decay beta
    Returns: None (updates the target model)
    """

    with torch.no_grad():
        param_dict_src = dict(model_src.named_parameters())

        for p_name, p_tgt in model_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

def adjust_dynamic_range(
    data: Tensor,
    drange_in: Optional[Tuple[float, float]] = (-1.0, 1.0),
    drange_out: Optional[Tuple[float, float]] = (0.0, 1.0),
):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0])
        )
        bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
        data = data * scale + bias

    return torch.clamp(data, min=drange_out[0], max=drange_out[1])


def post_process_generated_images(imgs: Tensor) -> np.array:
    imgs = adjust_dynamic_range(
        imgs.permute(0, 2, 3, 1), drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )
    return (imgs * 255.0).detach().cpu().numpy().astype(np.uint8)