import os
import tempfile
from pathlib import Path
import shutil
import cv2
from PIL import Image
from typing import Dict, List, Optional
from datetime import datetime

class VideoProcessor:
    def __init__(self, output_path: str, message_manager):
        self.output_path = output_path
        self.message_manager = message_manager
        self.temp_dir = None

    def get_video_info(self, video_path: str) -> Dict:
        """Get video information including frame count, fps, etc."""
        try:
            cap = cv2.VideoCapture(video_path)
            info = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            }
            cap.release()
            return info
        except Exception as e:
            self.message_manager.add_error(f"Error getting video info: {str(e)}")
            return {}

    def display_video_info(self, video_path: str) -> Optional[int]:
        """Display video information and return frame count"""
        info = self.get_video_info(video_path)
        if info:
            self.message_manager.add_message(
                f"Video Info:\n"
                f"• Resolution: {info['width']}x{info['height']}\n"
                f"• Frames: {info['frame_count']}\n"
                f"• FPS: {info['fps']:.2f}"
            )
            return info['frame_count']
        return None

    def prepare_video_processing(self, video_file) -> tuple[str, str, str]:
        """Create necessary directories for video processing"""
        # Use cleaned filename for folder name
        safe_name = Path(os.path.basename(video_file.name)).stem
        # Remove any problematic characters
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_folder = os.path.join(self.output_path, f"video_{safe_name}_{timestamp}")
        frames_in = os.path.join(video_folder, "frames_in")
        frames_out = os.path.join(video_folder, "frames_out")
        
        os.makedirs(video_folder, exist_ok=True)
        os.makedirs(frames_in, exist_ok=True)
        os.makedirs(frames_out, exist_ok=True)
        
        return video_folder, frames_in, frames_out

    def extract_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extract frames from video"""
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Save frame
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            pil_image.save(frame_path, "PNG")
            frame_paths.append(frame_path)
            
        cap.release()
        return frame_paths

    def reassemble_video(self, frames_dir: str, output_path: str, original_video: str) -> None:
        """Reassemble processed frames into video"""
        # Get original video info
        info = self.get_video_info(original_video)
        
        # Get first frame to determine dimensions
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if not frame_files:
            raise ValueError("No frames found to reassemble")
            
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            info['fps'],
            (width, height)
        )
        
        # Write frames
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)
            
        out.release()
        self.message_manager.add_success(f"Video saved to: {output_path}")
