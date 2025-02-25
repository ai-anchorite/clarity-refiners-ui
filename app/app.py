import os
import sys
import gradio as gr
import pillow_heif
import torch
import devicetorch
import subprocess
import tempfile
import gc
import psutil  # for system stats - gpu/cpu etc
import random
import shutil
import threading

from PIL import Image
from typing import List, Optional
from pathlib import Path
from datetime import datetime
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoProcessor

from refiners.foundationals.latent_diffusion import Solver, solvers
from enhancer import ESRGANUpscaler, ESRGANUpscalerCheckpoints

from system_monitor import SystemMonitor
from message_manager import MessageManager
from video_processor import VideoProcessor
from refiners.fluxion.utils import manual_seed  # Add this import

import warnings
# # Filter out the timm deprecation warning
# warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
# # Filter the GenerationMixin inheritance warning
# warnings.filterwarnings("ignore", message=".*has generative capabilities.*")
# # Filter the PyTorch flash attention warning
# warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

pillow_heif.register_heif_opener()
pillow_heif.register_avif_opener()

message_manager = MessageManager()

last_seed = None
save_path = "../outputs"   # Can be changed to a preferred directory: "C:\path\to\save_folder"
os.makedirs(save_path, exist_ok=True)
MAX_GALLERY_IMAGES = 30
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif'}


CHECKPOINTS = ESRGANUpscalerCheckpoints(
    unet=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.unet",
            filename="model.safetensors",
            revision="347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
        )
    ),
    clip_text_encoder=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.text_encoder",
            filename="model.safetensors",
            revision="744ad6a5c0437ec02ad826df9f6ede102bb27481",
        )
    ),
    lda=Path(
        hf_hub_download(
            repo_id="refiners/juggernaut.reborn.sd1_5.autoencoder",
            filename="model.safetensors",
            revision="3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
        )
    ),
    controlnet_tile=Path(
        hf_hub_download(
            repo_id="refiners/controlnet.sd1_5.tile",
            filename="model.safetensors",
            revision="48ced6ff8bfa873a8976fa467c3629a240643387",
        )
    ),
    # esrgan=Path(
        # hf_hub_download(
            # repo_id="philz1337x/upscaler",
            # filename="4x-UltraSharp.pth",
            # revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
        # )
    # ),
    negative_embedding=Path(
        hf_hub_download(
            repo_id="philz1337x/embeddings",
            filename="JuggernautNegative-neg.pt",
            revision="203caa7e9cc2bc225031a4021f6ab1ded283454a",
        )
    ),
    negative_embedding_key="string_to_param.*",
    
    loras={
        "more_details": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="more_details.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
        "sdxl_render": Path(
            hf_hub_download(
                repo_id="philz1337x/loras",
                filename="SDXLrender_v2.0.safetensors",
                revision="a3802c0280c0d00c2ab18d37454a8744c44e474e",
            )
        ),
    },
    esrgan_models={
        "4x-UltraSharp": Path(
            hf_hub_download(
                repo_id="philz1337x/upscaler",
                filename="4x-UltraSharp.pth",
                revision="011deacac8270114eb7d2eeff4fe6fa9a837be70",
            )
        ),
        "4x_foolhardy_Remacri": Path(
            hf_hub_download(
                repo_id="philz1337x/upscaler",
                filename="4x_foolhardy_Remacri.pth", 
                revision="076ac32b267863e0f86d11a05109b7b6cfd2de68",
            )
        ),
        "4x_NMKD-Superscale-SP_178000_G": Path(
            hf_hub_download(
                repo_id="gemasai/4x_NMKD-Superscale-SP_178000_G",
                filename="4x_NMKD-Superscale-SP_178000_G.pth", 
                revision="ecf220ff30d91515a97b7b9154badcf5c76323ef",
            )
        ),
        "4x_NMKD-Siax_200k": Path(
            hf_hub_download(
                repo_id="gemasai/4x_NMKD-Siax_200k",
                filename="4x_NMKD-Siax_200k.pth", 
                revision="6a21e0d6db5fe699873c9d1a6ae05c7dfe038171",
            )
        ),
        "x1_ITF_SkinDiffDetail_Lite_v1": Path(
            hf_hub_download(
                repo_id="gemasai/x1_ITF_SkinDiffDetail_Lite_v1",
                filename="x1_ITF_SkinDiffDetail_Lite_v1.pth", 
                revision="d96852260dcd8b39a0a4db62d674a81a2604adab",
            )
        ),
        # "4xFaceUpDAT": Path(
            # hf_hub_download(
                # repo_id="Phips/4xFaceUpDAT",
                # filename="4xFaceUpDAT.safetensors",
                # revision="21e1b10f8edf91425b66c7ad953f35226fed4b26",
            # )
        # ),
        # Add other models here
    },
)

device = torch.device(devicetorch.get(torch))
dtype = devicetorch.dtype(torch, "bfloat16")
enhancer = ESRGANUpscaler(checkpoints=CHECKPOINTS, device=device, dtype=dtype)


def generate_prompt(image: Image.Image, caption_detail: str = "<CAPTION>") -> str:
    """
    Generate a detailed caption for the image using Florence-2.
    """
    if image is None:
        message_manager.add_warning("No image loaded for captioning")
        return gr.Warning("Please load an image first!")
        
    try:
        message_manager.add_message(f"Starting Florence-2 caption generation with detail level: {caption_detail}")
        device = torch.device(devicetorch.get(torch))
        torch_dtype = devicetorch.dtype(torch, "bfloat16")
        
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Loading Florence-2 model...")

        # Load model in eval mode immediately
        model = AutoModelForCausalLM.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn", 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            "multimodalart/Florence-2-large-no-flash-attn",
            trust_remote_code=True
        )
        message_manager.add_success("Florence-2 model loaded successfully")

        # Move model to device after eval mode
        model = devicetorch.to(torch, model)
        message_manager.add_message("Processing image with Florence-2...")

        # Process the image
        inputs = processor(
            text=caption_detail, 
            images=image.convert("RGB"), 
            return_tensors="pt"
        )
        
        # Convert inputs to the correct dtype and move to device
        inputs = {
            k: v.to(device=device, dtype=torch_dtype if v.dtype == torch.float32 else v.dtype) 
            for k, v in inputs.items()
        }

        # Generate caption with no grad
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=2
            )
            
            # Move generated_ids to CPU immediately
            generated_ids = generated_ids.cpu()

        # Clear inputs from GPU
        inputs = {k: v.cpu() for k, v in inputs.items()}
        devicetorch.empty_cache(torch)
        
        # Process the generated text
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=caption_detail,
            image_size=(image.width, image.height)
        )
        
        # Clean up the caption and add enhancement-specific terms
        raw_caption = parsed_answer[caption_detail]
        caption_text = clean_caption(raw_caption)
        enhanced_prompt = f"masterpiece, best quality, highres, {caption_text}"
        
        message_manager.add_message("Raw caption: " + raw_caption)
        message_manager.add_success(f"Generated prompt: {enhanced_prompt}")

        # Aggressive cleanup
        del generated_ids
        del inputs
        model.cpu()
        del model
        del processor
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleaned up Florence-2 resources")
            
        return enhanced_prompt
        
    except Exception as e:
        # Ensure cleanup happens even on error
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_error(f"Error in caption generation: {str(e)}")
        return gr.Warning(f"Error generating prompt: {str(e)}")
        
        
def clean_caption(text: str) -> str:
    """
    Clean up the caption text by removing common prefixes, filler phrases, and dangling descriptions.
    """
    # Common prefixes to remove
    replacements = [
        "The image shows ",
        "The image is ",
        "The image depicts ",
        "This image shows ",
        "This image depicts ",
        "The photo shows ",
        "The photo depicts ",
        "The picture shows ",
        "The picture depicts ",
        "The overall mood ",
        "The mood of the image ",
        "There is ",
        "We can see ",
    ]
    
    cleaned_text = text
    for phrase in replacements:
        cleaned_text = cleaned_text.replace(phrase, "")
    
    # Remove mood/atmosphere fragments
    mood_patterns = [
        ". The mood is ",
        ". The atmosphere is ",
        ". of the image is ",
        ". The overall feel is ",
        ". The tone is ",
    ]
    
    for pattern in mood_patterns:
        if pattern in cleaned_text:
            cleaned_text = cleaned_text.split(pattern)[0]
    
    # Remove trailing fragments
    while cleaned_text.endswith((" is", " are", " and", " with", " the")):
        cleaned_text = cleaned_text.rsplit(" ", 1)[0]
    
    return cleaned_text.strip()


def get_seed(seed_value: int, reuse: bool) -> int:
    """Handle seed generation and reuse logic."""
    global last_seed
    
    if reuse and last_seed is not None:
        message_manager.add_message(f"Reusing previous seed: {last_seed}")
        return last_seed
    
    if seed_value == -1:
        generated_seed = random.randint(0, 2**32 - 1)
        last_seed = generated_seed
        message_manager.add_message(f"Generated random seed: {generated_seed}")
        return generated_seed
    
    last_seed = seed_value
    message_manager.add_message(f"Using provided seed: {seed_value}")
    return seed_value

    
def process(
    input_image: Image.Image,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    reuse_seed: bool = False,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
    auto_save_enabled: bool = True,  
    downscale_image: bool = True,
    downscale_size: int = 768,
    upscaler_model: str = "4x-UltraSharp",
) -> tuple[Image.Image, Image.Image]:
    try:
        # Input validation
        if input_image is None:
            message_manager.add_warning("No image loaded for enhancement")
            return gr.Warning("Please load an image first!")
            
        actual_seed = get_seed(seed, reuse_seed)
        message_manager.add_message(f"Starting enhancement with seed {actual_seed}")
        message_manager.add_message(f"Upscale factor: {upscale_factor}x")
        
        # Clear memory before processing
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleared GPU memory")

        solver_type: type[Solver] = getattr(solvers, solver)

        enhancer.set_model(upscaler_model)

        # Adjust inference steps if creativity is 0
        actual_steps = 1 if denoise_strength <= 0 else num_inference_steps
        message_manager.add_message(f"Inference steps: {actual_steps}")

        with torch.no_grad():
            message_manager.add_message("Processing image...")
            enhanced_image = enhancer.upscale(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                upscale_factor=upscale_factor,
                controlnet_scale=controlnet_scale,
                controlnet_scale_decay=controlnet_decay,
                condition_scale=condition_scale,
                tile_size=(tile_height, tile_width),
                denoise_strength=denoise_strength,
                num_inference_steps=actual_steps,  
                loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                solver_type=solver_type,
                seed=actual_seed, 
                downscale_image=downscale_image,
                downscale_size=downscale_size,
            )

        global latest_result
        latest_result = enhanced_image
        message_manager.add_success("Enhancement complete!")
        
        if auto_save_enabled:
            save_output(enhanced_image, True)
        
        # Clear memory after processing
        gc.collect()
        devicetorch.empty_cache(torch)
        message_manager.add_message("Cleaned up resources")
        
        return (input_image, enhanced_image)
        
    except Exception as e:
        message_manager.add_error(f"Error during processing: {str(e)}")
        gc.collect()
        devicetorch.empty_cache(torch)
        return gr.Warning(f"Error during processing: {str(e)}")

        
def batch_process_images(
    files,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = -1,
    reuse_seed: bool = False,
    upscale_factor: int = 2,
    controlnet_scale: float = 0.6,
    controlnet_decay: float = 1.0,
    condition_scale: int = 6,
    tile_width: int = 112,
    tile_height: int = 144,
    denoise_strength: float = 0.35,
    num_inference_steps: int = 18,
    solver: str = "DDIM",
    upscaler_model: str = "4x-UltraSharp",  # Add upscaler model parameter
    progress=gr.Progress()
) -> tuple[str, List[str], tuple[Image.Image, Image.Image]]:
    """
    Process multiple images with the enhancer and save directly to batch subfolder.
    """
    def generate_summary():
        """Helper to generate batch processing summary"""
        summary = [
            f"Processing complete!",
            f"Successfully processed: {results['successful']} images",
            f"Failed: {results['failed']} images",
            f"Skipped: {results['skipped']} images",
            f"\nSaved to folder: {batch_folder}"
        ]
        
        if results['error_files']:
            summary.append("\nErrors:")
            summary.extend(results['error_files'])
            
        return "\n".join(summary)

    if not files:
        message_manager.add_warning("No files selected for batch processing")
        return "Please upload some images to process.", [], (None, None)
        
    results = {
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'processed_files': [],
        'error_files': []
    }
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif', '.avif'}
    
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    batch_folder = os.path.join(save_path, f"batch_{timestamp}")
    os.makedirs(batch_folder, exist_ok=True)
    message_manager.add_message(f"Created batch folder: {batch_folder}")
    
    current_image_pair = (None, None)
    
    try:
        total_files = len(files)
        message_manager.add_message(f"Starting batch processing of {total_files} files")
        
        # Set initial seed for batch
        actual_seed = get_seed(seed, reuse_seed)
        
        # Set model at start of batch
        enhancer.set_model(upscaler_model)
        
        for i, file in enumerate(files, 1):
            try:
                # Update progress
                progress(i/total_files, f"Processing {i}/{total_files}")
                message_manager.add_message(f"Processing file {i}/{total_files}: {file.name}")
                
                # Check file extension
                file_ext = os.path.splitext(file.name)[1].lower()
                if file_ext not in valid_extensions:
                    message_manager.add_warning(f"Skipping unsupported file: {file.name}")
                    results['skipped'] += 1
                    results['error_files'].append(f"{os.path.basename(file.name)} (Unsupported format)")
                    
                    if i == total_files:
                        message_manager.add_success("Batch processing completed")
                        yield generate_summary(), update_gallery(), current_image_pair
                    else:
                        yield (
                            f"Processing {i}/{total_files}: {file.name}",
                            update_gallery(),
                            current_image_pair
                        )
                    continue
                
                # Load and process image
                input_image = Image.open(file.name).convert("RGB")
                
                # Clear memory before processing
                gc.collect()
                devicetorch.empty_cache(torch)
                
                # Update seed for each image if not reusing
                if not reuse_seed:
                    actual_seed = get_seed(-1, False)  # Generate new random seed
                
                message_manager.add_message(f"Processing with seed: {actual_seed}")
                
                solver_type: type[Solver] = getattr(solvers, solver)
                
                with torch.no_grad():
                    enhanced_image = enhancer.upscale(
                        image=input_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        upscale_factor=upscale_factor,
                        controlnet_scale=controlnet_scale,
                        controlnet_scale_decay=controlnet_decay,
                        condition_scale=condition_scale,
                        tile_size=(tile_height, tile_width),
                        denoise_strength=denoise_strength,
                        num_inference_steps=num_inference_steps,
                        loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                        solver_type=solver_type,
                        seed=actual_seed,
                        downscale_image=True,  # Add downscaling parameters
                        downscale_size=768,
                    )
                
                # Update the current image pair for the slider
                current_image_pair = (input_image, enhanced_image)
                
                # Save enhanced image to batch folder
                original_name = Path(file.name).stem
                enhanced_filename = f"{original_name}_enhanced.png"
                output_path = os.path.join(batch_folder, enhanced_filename)
                enhanced_image.save(output_path, "PNG")
                
                # Update results
                results['successful'] += 1
                results['processed_files'].append(enhanced_filename)
                message_manager.add_success(f"Saved: {enhanced_filename}")
                
                # For the last file, show the summary instead of progress
                if i == total_files:
                    message_manager.add_success("Batch processing completed")
                    yield generate_summary(), update_gallery(), current_image_pair
                else:
                    yield (
                        f"Processing {i}/{total_files}: {file.name}",
                        update_gallery(),
                        current_image_pair
                    )
                
                # Cleanup
                gc.collect()
                devicetorch.empty_cache(torch)
                
            except Exception as e:
                message_manager.add_error(f"Error processing {file.name}: {str(e)}")
                results['failed'] += 1
                results['error_files'].append(f"{os.path.basename(file.name)} ({str(e)})")
                
                if i == total_files:
                    message_manager.add_success("Batch processing completed")
                    yield generate_summary(), update_gallery(), current_image_pair
        
        # Final return for gradio
        return generate_summary(), update_gallery(), current_image_pair
        
    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}"
        message_manager.add_error(error_msg)
        return error_msg, [], (None, None)
            
            
def open_output_folder() -> None:
    folder_path = os.path.abspath(save_path)
    
    try:
        os.makedirs(folder_path, exist_ok=True)
    except Exception as e:
        message_manager.add_error(f"Error creating folder: {str(e)}")
        return
        
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', folder_path])
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
        message_manager.add_success(f"Opened outputs folder: {folder_path}")
    except Exception as e:
        message_manager.add_error(f"Error opening folder: {str(e)}")


def save_output(image: Image.Image = None, auto_saved: bool = False) -> List[str]:
    """Save image and return updated gallery data"""
    if image is None:
        if not globals().get('latest_result'):
            message_manager.add_warning("No image to save! Please enhance an image first.")
            return []
        image = latest_result
        
    try:
        # Create output directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        
        # Save the image
        image.save(filepath, "PNG")
        
        save_type = "auto-saved" if auto_saved else "saved"
        message = f"Image {save_type} as: {filename}"
        message_manager.add_success(message)
        
        # Return updated gallery data
        return update_gallery()
        
    except Exception as e:
        error_msg = f"Error saving image: {str(e)}"
        message_manager.add_error(error_msg)
        return []
        
        
def process_and_update(*args):
    """Wrapper to handle both process output and gallery update"""
    result = process(*args)  # This gives us the slider images
    return result, update_gallery()  # Get current gallery state
    
    
def update_gallery() -> List[str]:
    """Update gallery with most recent images from save path and batch folders."""
    try:
        # Get all images from main save path and batch subfolders
        batch_folders = [d for d in os.listdir(save_path) 
                        if os.path.isdir(os.path.join(save_path, d)) 
                        and d.startswith('batch_')]
        
        # Collect images from main folder and all batch folders
        all_images = []
        
        # Main folder images
        main_images = [
            os.path.join(save_path, f) 
            for f in os.listdir(save_path) 
            if f.lower().endswith(tuple(VALID_EXTENSIONS))
        ]
        all_images.extend(main_images)
        
        # Batch folder images
        for batch_folder in batch_folders:
            folder_path = os.path.join(save_path, batch_folder)
            batch_images = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(tuple(VALID_EXTENSIONS))
            ]
            all_images.extend(batch_images)
        
        # Sort by newest first and limit
        all_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return all_images[:MAX_GALLERY_IMAGES]
        
    except Exception as e:
        message_manager.add_error(f"Gallery update error: {str(e)}")
        return []


# Video work-in-progress

stop_video_processing = threading.Event()

def stop_video():
    """Stop video processing"""
    stop_video_processing.set()
    message_manager.add_message("Stopping video processing...")
    return "Stopping... Please wait for current frame to complete."

def update_video_info(video_file):
    """Display video information when a file is uploaded"""
    if video_file is None:
        return
        
    processor = VideoProcessor("../outputs", message_manager)
    try:
        frame_count = processor.display_video_info(video_file.name)
        if frame_count is not None:
            processing_msg = f"\n🎬 Ready to process {frame_count} frames"
            message_manager.add_message(processing_msg)
            return processing_msg
    except Exception as e:
        message_manager.add_error(f"Error processing video info: {str(e)}")
        return None
        
def process_video_and_update(
    video_file,
    prompt,
    negative_prompt,
    seed,
    reuse_seed,
    upscale_factor,
    controlnet_scale,
    controlnet_decay,
    condition_scale,
    tile_width,
    tile_height,
    denoise_strength,
    num_inference_steps,
    solver,
    upscaler_model, 
    progress=gr.Progress()
):
    """
    Gradio wrapper for video processing
    """
    # Reset stop flag at start
    stop_video_processing.clear()

    if video_file is None:
        message_manager.add_warning("No video file loaded")
        return None, update_gallery(), "Please load a video file first!"
        
    processor = VideoProcessor("../outputs", message_manager)
    
    try:
        # Prepare folders
        video_folder, frames_in, frames_out = processor.prepare_video_processing(video_file)
        
        # Get frame count for progress
        total_frames = processor.get_video_info(video_file.name)['frame_count']
        progress(0, desc="Starting video processing...")

        # Set ESRGAN model
        enhancer.set_model(upscaler_model)
        
        # Extract frames
        message_manager.add_message("Extracting frames...")
        frame_paths = processor.extract_frames(video_file.name, frames_in)
        progress(0.1, desc="Frames extracted")
        
        # Process frames
        solver_type = getattr(solvers, solver)
        actual_seed = get_seed(seed, reuse_seed)
        manual_seed(actual_seed)  # Now this will work
        
        # Calculate progress increment per frame
        frame_progress = 0.8 / total_frames  # 80% of progress bar for frame processing
        
        # Initial status update
        status = f"Processing {total_frames} frames..."
        current_image_pair = (None, None)
        
        for i, frame_path in enumerate(frame_paths, 1):
            # Check stop flag
            if stop_video_processing.is_set():
                message_manager.add_warning("Video processing stopped by user")
                yield current_image_pair, update_gallery(), "Processing stopped by user"
                return

            input_image = Image.open(frame_path).convert("RGB")
            gc.collect()
            devicetorch.empty_cache(torch)
            
            current_progress = 0.1 + (i * frame_progress)
            progress(current_progress, desc=f"Processing frame {i}/{total_frames}")
            message_manager.add_message(f"Processing frame {i}/{total_frames}")
            status = f"Processing frame {i}/{total_frames}"
            
            # Update seed for each frame if not reusing
            if not reuse_seed:
                actual_seed = get_seed(-1, False)
                
            with torch.no_grad():
                enhanced_frame = enhancer.upscale(
                    image=input_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    upscale_factor=upscale_factor,
                    controlnet_scale=controlnet_scale,
                    controlnet_scale_decay=controlnet_decay,
                    condition_scale=condition_scale,
                    tile_size=(tile_height, tile_width),
                    denoise_strength=denoise_strength,
                    num_inference_steps=num_inference_steps,
                    loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
                    solver_type=solver_type,
                    seed=actual_seed,
                    downscale_image=True,
                    downscale_size=768,
                )
            
            output_path = os.path.join(frames_out, f"frame_{i:04d}_enhanced.png")
            enhanced_frame.save(output_path, "PNG")
            
            current_image_pair = (input_image, enhanced_frame)
            # Update all outputs: slider, gallery, and status
            yield current_image_pair, update_gallery(), status
        
        # Reassemble video
        progress(0.9, desc="Reassembling video...")
        status = "Reassembling video..."
        message_manager.add_message("Reassembling video...")
        yield current_image_pair, update_gallery(), status
        
        output_video = os.path.join(video_folder, f"enhanced_video.mp4")
        processor.reassemble_video(frames_out, output_video, video_file.name)
        
        progress(1.0, desc="Complete!")
        final_status = f"Video processing complete! Saved to: {output_video}"
        message_manager.add_success(final_status)
        
        # Final yield with completion status
        yield current_image_pair, update_gallery(), final_status
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        message_manager.add_error(error_msg)
        yield None, update_gallery(), error_msg

        
css = """

/* Specific adjustments for Image */
.image-container .image-custom {
    max-width: 100% !important;
    max-height: 80vh !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    object-position: center !important;
}

/* Center the ImageSlider container and maintain full width for slider */
.image-container .image-slider-custom {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

/* Style for the slider container */
.image-container .image-slider-custom > div {
    width: 100% !important;
    max-width: 100% !important;
    max-height: 80vh !important;
}

/* Ensure both before/after images maintain aspect ratio */
.image-container .image-slider-custom img {
    max-height: 80vh !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

/* Style for the slider handle */
.image-container .image-slider-custom .image-slider-handle {
    width: 2px !important;
    background: white !important;
    border: 2px solid rgba(0, 0, 0, 0.6) !important;
}

.console-scroll textarea {
    max-height: 12em !important;  /* Approximately 8 lines of text */
    overflow-y: auto !important;  /* Enables vertical scrolling */
}

/* Status specific styling */
.batch-status textarea {
    min-height: 12em !important;  /* Ensures minimum height for welcome message */
}

.prompt-guide textarea {
    white-space: pre-wrap !important;  /* Preserves formatting but allows wrapping */
    padding-left: 1em !important;      /* Base padding for all text */
    text-indent: -1em !important;      /* Negative indent for first line */
}

"""

# Store the latest processing result
latest_result = None

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(elem_classes="image-container"):
            with gr.Tabs() as tabs:
                with gr.TabItem("Single Image") as single_tab:
                    input_image = gr.Image(type="pil", label="Input Image", elem_classes=["image-custom"])
                    run_button = gr.ClearButton(
                        components=None,
                        value="Enhance Image",
                        variant="primary"
                    )
                with gr.TabItem("Batch Process") as batch_tab:
                    batch_welcome = """✨ Welcome to Batch Processing! ✨

📸 Drop multiple images to enhance them sequentially:
    
• All enhancement settings will be applied to every image
   (prompts, denoise, seed, etc. - note: seed defaults to random)
• Images will be saved to a timestamped batch folder
• Enhancements appear in the before/after window
• Track progress here + additional details in main console

🚀 Ready? Drop your images above and click 'Process Batch'!"""

                    input_files = gr.File(
                        file_count="multiple",
                        label="Load Images",
                        scale=2
                    )
                    batch_status = gr.Textbox(
                        label="Batch Processing Status",
                        value=batch_welcome,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                        elem_classes="batch-status"
                    )
                    batch_button = gr.Button(
                        "Process Batch",
                        variant="primary"
                    )
                with gr.TabItem("Video Upscale") as video_tab:
                    video_welcome = """✨ Welcome to Video Processing! ✨

                📽️ Drop a video file to enhance it frame by frame:
                    
                • Video will be split into frames and enhanced
                • All enhancement settings will be applied to each frame
                • Progress appears in the before/after slider
                • Enhanced video saved to outputs folder
                • Track progress in main console

                ⚠️ Experimental Feature:
                • Currently limited to short test videos
                • Processing may take some time
                • No audio handling yet

                🎬 Ready? Drop a video file and click 'Process Video'!"""

                    input_video = gr.File(
                        file_count="single",
                        label="Load Video",
                        file_types=["video"],
                    )
                    video_status = gr.Textbox(
                        label="Video Processing Status",
                        value=video_welcome,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True,
                        elem_classes="batch-status"
                    )
                    with gr.Row():
                        video_button = gr.Button("Process Video", variant="primary")
                        stop_button = gr.Button("Stop Processing", variant="stop")
                    
        with gr.Column(elem_classes="image-container"):
            output_slider = ImageSlider(
                interactive=False,
                label="Before / After",
                elem_classes=["image-slider-custom"]
            )
            run_button.add(output_slider)
            with gr.Row():
                save_result = gr.Button("Save Result", scale=2)
                auto_save = gr.Checkbox(label="Auto-save", value=True)
                open_folder_button = gr.Button("Open Outputs Folder", scale=2)

    with gr.Accordion("Prompting", open=False):
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("Prompt"):
                    prompt = gr.Textbox(
                        label="Enhancement Prompt",
                        value="masterpiece, best quality, highres",
                        show_label=True
                    )
                with gr.TabItem("Guide"):
                    prompt_guide = gr.Textbox(
    value="""💡 Prompt Guide

🎯 Additional Prompts are optional! 
• They'll work similarly to prompts added to img2img with controlnets in other gen AI apps
• The default settings work great for general enhancement
• Use prompting to guide the AI towards specific improvements
• Keep prompts simple and focused on what you want enhanced

📝 Example prompts:
• "sharp details, high quality" - for clarity and definition
• "vivid colors, high contrast" - for more vibrant results
• "soft lighting, smooth details" - for a gentler enhancement
• "perfect eyes", "green eyes", detailed fingernails" etc - focus on specific details

💭 Tips:
• Start with the default prompt, then add specific guidance if needed.
• Florence2 auto-prompting is entirely optional. Mostly added because why not 😆""", 

                        label="Using Prompts",
                        interactive=False,
                        show_label=False,
                        lines=12,
                        elem_classes="prompt-guide"
                    )
            with gr.Column(scale=1):
                caption_detail = gr.Radio(
                    choices=["<CAPTION>","<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    value="<CAPTION>",
                    label="Florence-2 Caption Detail",
                    info="Choose level of detail for image analysis"
                )
                generate_prompt_btn = gr.Button("📝 Generate Prompt", variant="primary")
        with gr.Row():
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="worst quality, low quality, normal quality",
            )
                
    with gr.Accordion("Options", open=True):
        with gr.Row():
            upscaler_model = gr.Dropdown(
                choices=list(CHECKPOINTS.esrgan_models.keys()),
                value="4x-UltraSharp",
                label="Upscaler Model"
            )
            downscale_image = gr.Checkbox(
                label="Auto-resize",
                value=True,
                info="Resize input image to target size before upscaling"
            )
            downscale_size = gr.Slider(
                minimum=512,
                maximum=2048,
                value=768,
                step=256,
                label="Target size for shortest side",
                interactive=True,
                visible=True
            )    
        with gr.Row(equal_height=True):    
            with gr.Column():
                upscale_factor = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=2,
                    step=0.5,
                    label="Upscale Factor",
                )
            with gr.Column():
                creativity = gr.Slider(  # Renamed from denoise_strength
                    minimum=0,
                    maximum=1,
                    value=0.15,
                    step=0.05,
                    label="Creativity", 
                    info="0 for pure upscaling, higher values add artistic enhancement",
                )
            with gr.Column():
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=20,
                    step=1,
                    label="Number of Inference Steps",
                )
        with gr.Row():
            seed = gr.Slider(
                label="Seed (-1 = rnd)",
                minimum=-1,
                maximum=2**32 - 1,
                step=1,
                value=-1,
                scale=4,
            )
            reuse_seed = gr.Checkbox(
                label="Reuse seed", 
                value=False,
                scale=1,
            )
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row(): 
            controlnet_scale = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.6,
                step=0.1,
                label="ControlNet Scale",
            )
            controlnet_decay = gr.Slider(
                minimum=0.5,
                maximum=1,
                value=1.0,
                step=0.025,
                label="ControlNet Scale Decay",
            )
            condition_scale = gr.Slider(
                minimum=2,
                maximum=20,
                value=6,
                step=1,
                label="Condition Scale",
            )
        with gr.Row(): 
            tile_width = gr.Slider(
                minimum=64,
                maximum=200,
                value=112,
                step=1,
                label="Latent Tile Width",
            )
            tile_height = gr.Slider(
                minimum=64,
                maximum=200,
                value=144,
                step=1,
                label="Latent Tile Height",
            )
            solver = gr.Radio(
                choices=["DDIM", "DPMSolver"],
                value="DDIM",
                label="Solver",
            )
    with gr.Accordion("System Info & Console", open=True):            
        with gr.Row():       
            # Status Info (for cpu/gpu monitor)
            resource_monitor = gr.Textbox(
                label="Monitor",
                lines=8,
                interactive=False,
                # value=get_welcome_message()
            )  
            console_out = gr.Textbox(
                label="Console",
                lines=8,
                interactive=False,
                show_copy_button=True,
                autoscroll=True,    # Enables automatic scrolling to newest messages
                elem_classes="console-scroll"  # Add custom class for styling
            )
 
    with gr.Accordion("Gallery", open=False):     
        with gr.Row():
            gallery = gr.Gallery(
                label="Recent Enhancements",
                show_label=True,
                elem_id="gallery",
                columns=5,
                rows=6,
                height="80vh",  # Use viewport height instead of fixed pixels
                object_fit="contain",
                allow_preview=True,
                show_share_button=False,
                show_download_button=True,
                preview=True,
            )
            
    # Event handlers
    
    generate_prompt_btn.click(
        fn=generate_prompt,
        inputs=[input_image, caption_detail],
        outputs=[prompt]
    )
    
    run_button.click(
        fn=process_and_update,
        inputs=[
            input_image,
            prompt,
            negative_prompt,
            seed,
            reuse_seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            creativity,
            num_inference_steps,
            solver,
            auto_save,
            downscale_image,
            downscale_size,
            upscaler_model,
        ],
        outputs=[output_slider, gallery]
    )
    
    batch_button.click(
        fn=batch_process_images,
        inputs=[
            input_files,
            prompt,
            negative_prompt,
            seed,
            reuse_seed,
            upscale_factor,
            controlnet_scale,
            controlnet_decay,
            condition_scale,
            tile_width,
            tile_height,
            creativity,
            num_inference_steps,
            solver,
            upscaler_model, 
        ],
        outputs=[batch_status, gallery, output_slider]
    )

    input_video.upload(
        fn=update_video_info,
        inputs=[input_video],
        outputs=[video_status]
    )
 
    video_button.click(
            fn=process_video_and_update,
            inputs=[
                input_video,
                prompt,
                negative_prompt,
                seed,
                reuse_seed,
                upscale_factor,
                controlnet_scale,
                controlnet_decay,
                condition_scale,
                tile_width,
                tile_height,
                creativity,
                num_inference_steps,
                solver,
                upscaler_model, 
            ],
            outputs=[output_slider, gallery, video_status] 
        )
        
    stop_button.click(
        fn=stop_video,
        inputs=None,
        outputs=[video_status],
    )
    
    save_result.click(
        fn=save_output,
        inputs=None,
        outputs=[gallery]
    )
    
    open_folder_button.click(
        fn=open_output_folder,
        inputs=None,
        outputs=None  # Remove the output entirely
    )
    
    def update_console():
        return message_manager.get_messages()
        
    def update_monitor():
        """Separate function for system monitoring to avoid folder opening"""
        return SystemMonitor.get_system_info(), message_manager.get_messages()
        
    # Initialize the timer and set up its tick event
    demo.load(
        fn=update_monitor,
        outputs=[resource_monitor, console_out],
        every=2
    )
    
    
demo.launch(share=False)
