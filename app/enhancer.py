from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
    MultiUpscaler,
    UpscalerCheckpoints,
)
from refiners.fluxion.utils import load_from_safetensors

# from dat_model import UpscalerDAT
from esrgan_model import UpscalerESRGAN


@dataclass(kw_only=True)
class ESRGANUpscalerCheckpoints(UpscalerCheckpoints):
    esrgan_models: Dict[str, Path]  # Changed to support multiple models

class ESRGANUpscaler(MultiUpscaler):
    def __init__(
        self,
        checkpoints: ESRGANUpscalerCheckpoints,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(checkpoints=checkpoints, device=device, dtype=dtype)
        self.esrgan_models = {}
        # Load all models, detecting type by extension
        for name, path in checkpoints.esrgan_models.items():
            if path.suffix == '.safetensors':
                # DAT model
                self.esrgan_models[name] = UpscalerDAT(path, device=self.device, dtype=self.dtype)
            else:
                # ESRGAN model
                self.esrgan_models[name] = UpscalerESRGAN(path, device=self.device, dtype=self.dtype)
        self.current_model = list(self.esrgan_models.keys())[0]  # Default to first model

    def to(self, device: torch.device, dtype: torch.dtype):
        for model in self.esrgan_models.values():
            model.to(device=device, dtype=dtype)
        self.sd = self.sd.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def set_model(self, model_name: str):
        """Switch active ESRGAN model"""
        if model_name in self.esrgan_models:
            self.current_model = model_name

    def upscale(self, image: Image.Image, **kwargs) -> Image.Image:
        """Override upscale to handle pure ESRGAN mode"""
        if kwargs.get('denoise_strength', 0.35) <= 0:
            # Pure ESRGAN mode
            esrgan = self.esrgan_models[self.current_model]
            scale_factor = esrgan.scale_factor
            upscaled = esrgan.upscale_with_tiling(image)
            target_scale = kwargs.get('upscale_factor', 2)
            
            if target_scale != scale_factor:
                # Resize to desired scale if not using native scale
                new_w = int(image.width * target_scale)
                new_h = int(image.height * target_scale)
                upscaled = upscaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return upscaled
        
        # Normal diffusion-enhanced mode
        return super().upscale(image, **kwargs)

    def pre_upscale(self, image: Image.Image, upscale_factor: float, **kwargs: Any) -> Image.Image:
        """Enhanced pre-upscale with ESRGAN models"""
        if kwargs.get('denoise_strength', 0.35) > 0:
            # Only do ESRGAN pre-processing for diffusion mode
            esrgan = self.esrgan_models[self.current_model]
            scale_factor = esrgan.scale_factor  # Get model's native scale
            image = esrgan.upscale_with_tiling(image)
            return super().pre_upscale(image=image, upscale_factor=upscale_factor / scale_factor)
        return image  # In pure mode, upscale handles everything
