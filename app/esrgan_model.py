"""
Modified from https://github.com/philz1337x/clarity-upscaler
which is a copy of https://github.com/AUTOMATIC1111/stable-diffusion-webui
which is a copy of https://github.com/victorca25/iNNfer
which is a copy of https://github.com/xinntao/ESRGAN
"""

import math
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from PIL import Image


def conv_block(in_nc: int, out_nc: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}
    """

    def __init__(self, nf: int = 64, gc: int = 32) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]

        self.conv1 = conv_block(nf, gc)
        self.conv2 = conv_block(nf + gc, gc)
        self.conv3 = conv_block(nf + 2 * gc, gc)
        self.conv4 = conv_block(nf + 3 * gc, gc)
        # Wrapped in Sequential because of key in state dict.
        self.conv5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, kernel_size=3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(self, nf: int) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.RDB1 = ResidualDenseBlock_5C(nf)
        self.RDB2 = ResidualDenseBlock_5C(nf)
        self.RDB3 = ResidualDenseBlock_5C(nf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class Upsample2x(nn.Module):
    """Upsample 2x."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=2.0)  # type: ignore


class ShortcutBlock(nn.Module):
    """Elementwise sum the output of a submodule to its input"""

    def __init__(self, submodule: nn.Module) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.sub = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(x)


class RRDBNet(nn.Module):
    def __init__(self, in_nc: int, out_nc: int, nf: int, nb: int, scale: int = 4) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        assert in_nc % 4 != 0  # in_nc is 3

        # Base layers
        modules = [
            nn.Conv2d(in_nc, nf, kernel_size=3, padding=1),
            ShortcutBlock(
                nn.Sequential(
                    *(RRDB(nf) for _ in range(nb)),
                    nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                )
            ),
        ]

        # Add upscale blocks for 2x and 4x models
        if scale >= 2:
            modules.extend([
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ])
        if scale >= 4:
            modules.extend([
                Upsample2x(),
                nn.Conv2d(nf, nf, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            ])

        # Final layers
        modules.extend([
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, out_nc, kernel_size=3, padding=1),
        ])

        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def infer_params(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, int]:
    # Calculate scale factor by checking model architecture
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    out_nc = 0
    nb = 0

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > scalemin and parts[0] == "model" and parts[2] == "weight":
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    scale = 2**scale2x if scale2x > 0 else 1  # If no upscale layers found, assume 1x

    assert out_nc > 0
    assert nb > 0

    print(f"Inferred model parameters: in_nc={in_nc}, out_nc={out_nc}, nf={nf}, nb={nb}, scale={scale}")
    return in_nc, out_nc, nf, nb, scale


Tile = tuple[int, int, Image.Image]
Tiles = list[tuple[int, int, list[Tile]]]


# https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L64
class Grid(NamedTuple):
    tiles: Tiles
    tile_w: int
    tile_h: int
    image_w: int
    image_h: int
    overlap: int


# adapted from https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L67
def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w = image.width
    h = image.height

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = max(1, math.ceil((w - overlap) / non_overlap_width))
    rows = max(1, math.ceil((h - overlap) / non_overlap_height))

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images: list[Tile] = []
        y1 = max(min(int(row * dy), h - tile_h), 0)
        y2 = min(y1 + tile_h, h)
        for col in range(cols):
            x1 = max(min(int(col * dx), w - tile_w), 0)
            x2 = min(x1 + tile_w, w)
            tile = image.crop((x1, y1, x2, y2))
            row_images.append((x1, tile_w, tile))
        grid.tiles.append((y1, tile_h, row_images))

    print(f"Split image {w}x{h} into {rows}x{cols} grid with tile size {tile_w}x{tile_h}")
    return grid


# https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/images.py#L104
def combine_grid(grid: Grid):
    def make_mask_image(r: npt.NDArray[np.float32]) -> Image.Image:
        r = r * 255 / grid.overlap
        return Image.fromarray(r.astype(np.uint8), "L")

    mask_w = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0)
    )
    mask_h = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1)
    )

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(
            combined_row.crop((0, 0, combined_row.width, grid.overlap)),
            (0, y),
            mask=mask_h,
        )
        combined_image.paste(
            combined_row.crop((0, grid.overlap, combined_row.width, h)),
            (0, y + grid.overlap),
        )

    print(f"Combined grid into image of size {combined_image.width}x{combined_image.height}")
    return combined_image


class UpscalerESRGAN:
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        self.model_path = model_path
        self.device = device
        print(f"Initializing ESRGAN model from {model_path}")
        
        try:
            # Get scale factor first
            state_dict = torch.load(model_path, weights_only=True, map_location="cpu")  # Load to CPU first
            print(f"Loaded state dict with {len(state_dict)} keys")
            
            _, _, _, _, self.scale_factor = infer_params(state_dict)
            print(f"Inferred scale factor: {self.scale_factor}")
            
            # Now load model with correct scale factor
            self.model = self.load_model(model_path)
            print(f"Model loaded successfully")
            self.to(device, dtype)
            
            # Verify model works with a simple test
            test_input = torch.randn(1, 3, 16, 16, device=device, dtype=dtype)
            with torch.no_grad():
                test_output = self.model(test_input)
            expected_size = 16 * self.scale_factor
            print(f"Model test: input 16x16 → output {test_output.shape[2]}x{test_output.shape[3]}")
            if test_output.shape[2] != expected_size or test_output.shape[3] != expected_size:
                print(f"WARNING: Model output size {test_output.shape[2]}x{test_output.shape[3]} doesn't match expected {expected_size}x{expected_size}")
        except Exception as e:
            print(f"Error initializing ESRGAN model: {str(e)}")
            traceback.print_exc()
            raise

    def __call__(self, img: Image.Image) -> Image.Image:
        return self.upscale_without_tiling(img)

    def to(self, device: torch.device, dtype: torch.dtype):
        print(f"Moving ESRGAN model to {device} with dtype {dtype}")
        self.device = device
        self.dtype = dtype
        self.model.to(device=device, dtype=dtype)

    def load_model(self, path: Path) -> RRDBNet:
        print(f"Loading ESRGAN model from {path}")
        filename = path
        state_dict = torch.load(filename, weights_only=True, map_location=self.device)
        in_nc, out_nc, nf, nb, upscale = infer_params(state_dict)
        # Pass scale factor to RRDBNet
        model = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, scale=upscale)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"ESRGAN model loaded with scale factor {upscale}")
        return model

    def upscale_without_tiling(self, img: Image.Image) -> Image.Image:
        print(f"Upscaling image without tiling: {img.width}x{img.height}")
        try:
            img_np = np.array(img)
            img_np = img_np[:, :, ::-1]  # RGB to BGR
            img_np = np.ascontiguousarray(np.transpose(img_np, (2, 0, 1))) / 255
            img_t = torch.from_numpy(img_np).float()  # type: ignore
            img_t = img_t.unsqueeze(0).to(device=self.device, dtype=self.dtype)
            
            print(f"Input tensor shape: {img_t.shape}, device: {img_t.device}, dtype: {img_t.dtype}")
            
            with torch.no_grad():
                output = self.model(img_t)
                
            print(f"Output tensor shape: {output.shape}")
            
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = 255.0 * np.moveaxis(output, 0, 2)
            output = output.astype(np.uint8)
            output = output[:, :, ::-1]  # BGR to RGB
            result = Image.fromarray(output, "RGB")
            
            print(f"Upscaled to {result.width}x{result.height}")
            return result
        except Exception as e:
            print(f"Error in upscale_without_tiling: {str(e)}")
            traceback.print_exc()
            raise

    # https://github.com/philz1337x/clarity-upscaler/blob/e0cd797198d1e0e745400c04d8d1b98ae508c73b/modules/esrgan_model.py#L208
    def upscale_with_tiling(self, img: Image.Image) -> Image.Image:
        print(f"Upscaling image with tiling: {img.width}x{img.height}")
        try:
            img = img.convert("RGB")
            grid = split_grid(img)
            newtiles: Tiles = []
            scale_factor: int = 1

            for y, h, row in grid.tiles:
                newrow: list[Tile] = []
                for tiledata in row:
                    x, w, tile = tiledata
                    print(f"Processing tile at ({x},{y}) with size {tile.width}x{tile.height}")
                    output = self.upscale_without_tiling(tile)
                    scale_factor = output.width // tile.width
                    print(f"Tile upscaled to {output.width}x{output.height} (scale factor: {scale_factor})")
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
            output = combine_grid(newgrid)
            print(f"Final combined output: {output.width}x{output.height}")
            return output
        except Exception as e:
            print(f"Error in upscale_with_tiling: {str(e)}")
            traceback.print_exc()
            raise


def test_esrgan_model(model_path, device=None):
    """Test function to verify ESRGAN model is working correctly"""
    from PIL import Image
    import numpy as np
    import torch
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple test image
    test_img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    # Add some patterns to make upscaling visible
    for i in range(64):
        for j in range(64):
            if (i + j) % 8 == 0:
                test_img.putpixel((i, j), (255, 0, 0))
            elif (i - j) % 8 == 0:
                test_img.putpixel((i, j), (0, 255, 0))
    
    print(f"Testing ESRGAN model: {model_path}")
    try:
        # Initialize the model
        upscaler = UpscalerESRGAN(model_path, device, torch.float32)
        
        # Test upscaling
        print("Upscaling test image...")
        result = upscaler.upscale_without_tiling(test_img)
        
        # Verify dimensions
        expected_w = test_img.width * upscaler.scale_factor
        expected_h = test_img.height * upscaler.scale_factor
        
        print(f"Original size: {test_img.width}x{test_img.height}")
        print(f"Expected size: {expected_w}x{expected_h}")
        print(f"Actual size: {result.width}x{result.height}")
        
        # Save test images
        test_img.save("esrgan_test_input.png")
        result.save("esrgan_test_output.png")
        print("Test images saved to esrgan_test_input.png and esrgan_test_output.png")
        
        return True
    except Exception as e:
        print(f"Test failed: {str(e)}")
        traceback.print_exc()
        return False

# Uncomment to run test when module is executed directly
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) > 1:
#         test_esrgan_model(Path(sys.argv[1]))
#     else:
#         print("Please provide path to ESRGAN model")
