import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from refiners.fluxion.utils import load_from_safetensors
from esrgan_model import split_grid, combine_grid, Grid  # Import tiling functions at module level

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, var, weight = ctx.saved_variables
        N, C, H, W = grad_output.size()
        g_y = grad_output * weight.view(1, C, 1, 1)
        g_mu = (g_y * -1).sum(dim=(2, 3), keepdim=True)
        g_var = (g_y * y * -0.5 * (var + eps).pow(-1.5)).sum(dim=(2, 3), keepdim=True)
        g_x = g_y / (var + eps).sqrt() + g_var * 2 * (y * weight.view(1, C, 1, 1)) / (N * H * W) + g_mu / (N * H * W)
        g_weight = (grad_output * y).sum(dim=(0, 2, 3), keepdim=True)
        g_bias = grad_output.sum(dim=(0, 2, 3), keepdim=True)
        return g_x, g_weight.squeeze(), g_bias.squeeze(), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class DATBlock(nn.Module):
    """Modified DATBlock to match FaceUpDAT architecture"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1),
            SimpleGate(),
            nn.Conv2d(2 * dim, dim, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DATUpscaler(nn.Module):
    def __init__(self, in_channels=3, dim=64, num_blocks=12, upscale=4):
        super().__init__()
        
        # Input conv
        self.fea_conv = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        
        # Body module
        modules_body = []
        for _ in range(num_blocks):
            modules_body.append(
                DATBlock(dim=dim, num_heads=8)
            )
        self.body = nn.Sequential(*modules_body)
        
        # Upsample module
        modules_tail = []
        if upscale == 4:
            modules_tail.append(
                nn.Sequential(
                    nn.Conv2d(dim, 4*dim, 3, 1, 1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(dim, 4*dim, 3, 1, 1),
                    nn.PixelShuffle(2)
                )
            )
        else:
            modules_tail.append(
                nn.Sequential(
                    nn.Conv2d(dim, upscale*upscale*dim, 3, 1, 1),
                    nn.PixelShuffle(upscale)
                )
            )
        modules_tail.append(nn.Conv2d(dim, 3, 3, 1, 1))
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        feat = self.fea_conv(x)
        body_out = self.body(feat)
        body_out = body_out + feat
        out = self.tail(body_out)
        return out

class UpscalerDAT:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
        # Load the model using safetensors
        if str(model_path).endswith('.safetensors'):
            state_dict = load_from_safetensors(model_path, device='cpu')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        
        # Rest stays the same
        self.scale_factor = self._infer_scale_factor(state_dict)
        self.model = self._build_model(state_dict)
        self.model.eval()
        self.to(device, dtype)

    def _infer_scale_factor(self, state_dict):
        # Analyze the state dict to determine upscale factor
        # This is model-specific and might need adjustment
        return 4  # Most DAT models are 4x
        
    def _build_model(self, state_dict):
        model = DATUpscaler(upscale=self.scale_factor)
        model.load_state_dict(state_dict)
        return model

    def to(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self.model.to(device=device, dtype=dtype)

    def upscale_without_tiling(self, img: Image.Image) -> Image.Image:
        # Convert to tensor
        img_np = np.array(img)
        img_np = img_np.transpose(2, 0, 1) / 255.0  # DAT expects [0,1] range
        img_t = torch.from_numpy(img_np).float().unsqueeze(0)
        img_t = img_t.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            output = self.model(img_t)
            
        # Convert back to image
        output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).transpose(1, 2, 0).astype(np.uint8)
        return Image.fromarray(output)

    def upscale_with_tiling(self, img: Image.Image) -> Image.Image:
        # Use the tiling functions imported at module level
        img = img.convert("RGB")
        grid = split_grid(img)
        newtiles = []
        scale_factor = 1

        for y, h, row in grid.tiles:
            newrow = []
            for tiledata in row:
                x, w, tile = tiledata
                output = self.upscale_without_tiling(tile)
                scale_factor = output.width // tile.width
                newrow.append((x * scale_factor, w * scale_factor, output))
            newtiles.append((y * scale_factor, h * scale_factor, newrow))

        newgrid = Grid(
            newtiles,
            grid.tile_w * scale_factor,
            grid.tile_h * scale_factor,
            grid.image_w * scale_factor,
            grid.image_h * scale_factor,
            grid.overlap * scale_factor,
        )
        return combine_grid(newgrid)
