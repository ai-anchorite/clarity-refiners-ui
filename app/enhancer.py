from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import traceback

import torch
from PIL import Image
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
    MultiUpscaler,
    UpscalerCheckpoints,
)
from refiners.fluxion.utils import load_from_safetensors

from dat_model import UpscalerDAT
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
        print(f"Loading upscaler models...")
        for name, path in checkpoints.esrgan_models.items():
            try:
                print(f"Loading model {name} from {path}")
                if path.suffix == '.safetensors':
                    # DAT model
                    self.esrgan_models[name] = UpscalerDAT(path, device=self.device, dtype=self.dtype)
                    print(f"Loaded DAT model {name}")
                else:
                    # ESRGAN model
                    self.esrgan_models[name] = UpscalerESRGAN(path, device=self.device, dtype=self.dtype)
                    print(f"Loaded ESRGAN model {name}")
            except Exception as e:
                print(f"Error loading model {name}: {str(e)}")
                traceback.print_exc()
                
        if self.esrgan_models:
            self.current_model = list(self.esrgan_models.keys())[0]  # Default to first model
            print(f"Set default model to {self.current_model}")
        else:
            print("WARNING: No upscaler models were loaded successfully!")

    def to(self, device: torch.device, dtype: torch.dtype):
        for name, model in self.esrgan_models.items():
            try:
                model.to(device=device, dtype=dtype)
                print(f"Moved model {name} to {device} with dtype {dtype}")
            except Exception as e:
                print(f"Error moving model {name} to device: {str(e)}")
        self.sd = self.sd.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def set_model(self, model_name: str):
        """Switch active ESRGAN model"""
        if model_name in self.esrgan_models:
            self.current_model = model_name
            print(f"Set active model to {model_name}")
        else:
            available_models = list(self.esrgan_models.keys())
            print(f"Model {model_name} not found. Available models: {available_models}")
            raise ValueError(f"Model {model_name} not found. Available models: {available_models}")

    def upscale(self, image: Image.Image, **kwargs) -> Image.Image:
        """Override upscale to handle pure ESRGAN mode"""
        denoise_strength = kwargs.get('denoise_strength', 0.35)
        print(f"Upscaling with denoise_strength={denoise_strength}")
        
        if denoise_strength <= 0:
            # Pure ESRGAN mode
            print(f"Using pure ESRGAN mode with model {self.current_model}")
            try:
                esrgan = self.esrgan_models[self.current_model]
                scale_factor = esrgan.scale_factor
                print(f"Model native scale factor: {scale_factor}")
                
                print(f"Input image size: {image.width}x{image.height}")
                upscaled = esrgan.upscale_with_tiling(image)
                print(f"After ESRGAN: {upscaled.width}x{upscaled.height}")
                
                target_scale = kwargs.get('upscale_factor', 2)
                print(f"Target scale: {target_scale}")
                
                if target_scale != scale_factor:
                    # Resize to desired scale if not using native scale
                    new_w = int(image.width * target_scale)
                    new_h = int(image.height * target_scale)
                    print(f"Resizing to target dimensions: {new_w}x{new_h}")
                    upscaled = upscaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                print(f"Final output size: {upscaled.width}x{upscaled.height}")
                return upscaled
            except Exception as e:
                print(f"Error in pure ESRGAN mode: {str(e)}")
                traceback.print_exc()
                raise
        
        # Normal diffusion-enhanced mode
        print("Using diffusion-enhanced mode")
        
        # Filter out kwargs that the parent class doesn't accept
        diffusion_kwargs = {k: v for k, v in kwargs.items() if k not in ['downscale_image', 'downscale_size']}
        
        # Handle downscaling if needed
        if kwargs.get('downscale_image', False):
            downscale_size = kwargs.get('downscale_size', 768)
            print(f"Downscaling image to {downscale_size} before diffusion processing")
            # Resize while maintaining aspect ratio
            w, h = image.size
            if w < h:
                new_w = downscale_size
                new_h = int(h * (downscale_size / w))
            else:
                new_h = downscale_size
                new_w = int(w * (downscale_size / h))
            
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            print(f"Downscaled to {image.width}x{image.height}")
        
        return super().upscale(image, **diffusion_kwargs)

    def pre_upscale(self, image: Image.Image, upscale_factor: float, **kwargs: Any) -> Image.Image:
        """Enhanced pre-upscale with ESRGAN models"""
        denoise_strength = kwargs.get('denoise_strength', 0.35)
        print(f"Pre-upscale with denoise_strength={denoise_strength}")
        
        if denoise_strength > 0:
            # Only do ESRGAN pre-processing for diffusion mode
            try:
                print(f"Using ESRGAN pre-processing with model {self.current_model}")
                esrgan = self.esrgan_models[self.current_model]
                scale_factor = esrgan.scale_factor  # Get model's native scale
                print(f"Model native scale factor: {scale_factor}")
                
                print(f"Input image size: {image.width}x{image.height}")
                image = esrgan.upscale_with_tiling(image)
                print(f"After ESRGAN pre-processing: {image.width}x{image.height}")
                
                adjusted_factor = upscale_factor / scale_factor
                print(f"Adjusted upscale factor for diffusion: {adjusted_factor}")
                
                result = super().pre_upscale(image=image, upscale_factor=adjusted_factor)
                print(f"After pre_upscale: {result.width}x{result.height}")
                return result
            except Exception as e:
                print(f"Error in ESRGAN pre-processing: {str(e)}")
                traceback.print_exc()
                # Fall back to standard pre-upscale
                return super().pre_upscale(image=image, upscale_factor=upscale_factor)
                
        print("Skipping pre-upscale in pure ESRGAN mode")
        return image  # In pure mode, upscale handles everything
