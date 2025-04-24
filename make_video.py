import os
import cv2
import argparse
import imageio  # Add imageio for GIF creation
from PIL import Image  # Replace imghdr with Pillow
from pathlib import Path
import numpy as np  # Import numpy for the codec detection
import pillow_heif  # Import pillow_heif for HEIC support
import subprocess  # For checking FFmpeg availability
import sys  # For progress bar output
import time  # For progress updates

# Register heif opener with pillow
pillow_heif.register_heif_opener()

# Define format extensions at module level so they can be reused
FORMAT_EXTENSIONS = {
    "mp4": ".mp4",
    "gif": ".gif",
    "webm": ".webm",
    "avi": ".avi",
    "mov": ".mov",
    "mkv": ".mkv",
    "wmv": ".wmv"
}

# Define codecs for OpenCV fallback (simpler approach)
OPENCV_FALLBACK_CODECS = {
    "mp4": "mp4v",
    "avi": "XVID",  # Or MJPG
    "mov": "mp4v",
    "wmv": "WMV2"  # Keep WMV as an option if explicitly requested
}

def print_progress(current, total, prefix='Progress:', suffix='Complete', length=50, fill='█', print_end='\r'):
    """
    Display a progress bar in the console
    
    Args:
        current (int): Current progress value
        total (int): Total value for 100% completion
        prefix (str): Text to display before the progress bar
        suffix (str): Text to display after the progress bar
        length (int): Character length of the progress bar
        fill (str): Bar fill character
        print_end (str): End character (e.g. '\r', '\n')
    """
    percent = f"{100 * (current / float(total)):.1f}%"
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent} {suffix}')
    sys.stdout.flush()
    # Print a newline when complete
    if current >= total:
        print()

def check_ffmpeg():
    """Check if FFmpeg is available in the system path"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, Exception):
        return False

def make_video_ffmpeg(image_folder, output_name, duration, images, resize_width, resize_height, durations):
    """Create video using FFmpeg if available"""
    if not check_ffmpeg():
        print("FFmpeg not found. Please install FFmpeg for better video compatibility.")
        return False
        
    try:
        # Create a temporary folder for frames
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_ffmpeg_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process all images with progress tracking
        print(f"Processing {len(images)} images for video creation...")
        total_frames = 0
        for idx, image in enumerate(images):
            # Calculate total frames including duplicates for duration
            frame_duration = next((d for i, d in durations if i == idx), duration)
            frame_count = int(frame_duration / duration)
            total_frames += frame_count
            
        processed_frames = 0
        for idx, image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = load_image(image_path)
            
            if frame is None:
                print(f"\nWarning: Could not read image: {image_path}. Skipping this frame.")
                continue
                
            # Resize frame if needed
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
                
            # Save frame to temp directory
            frame_path = os.path.join(temp_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(frame_path, frame)
            processed_frames += 1
            
            # Handle frame durations
            frame_duration = next((d for i, d in durations if i == idx), duration)
            if frame_duration > duration:
                # For longer durations, create duplicate frames
                factor = int(frame_duration / duration) - 1
                for dup in range(1, factor+1):
                    dup_path = os.path.join(temp_dir, f"frame_{idx:05d}_{dup}.png")
                    cv2.imwrite(dup_path, frame)
                    processed_frames += 1
            
            # Update progress
            print_progress(processed_frames, total_frames, prefix='Preparing Frames:', suffix='Complete')
        
        print("\nEncoding video with FFmpeg...")
        
        # Use FFmpeg to create video with Windows-compatible settings
        fps = 1 / duration
        
        # Choose format-specific settings for maximum Windows compatibility
        format_ext = os.path.splitext(output_name)[1].lower()
        
        # Base command structure
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", os.path.join(temp_dir, "*.png"),
        ]

        # Format-specific codec and options
        if format_ext == '.mp4':
            # Use Main profile and Level 3.1 for broader WMP compatibility
            print("Using FFmpeg with Main Profile Level 3.1 for MP4 (better WMP compatibility)")
            cmd.extend([
                "-c:v", "libx264",
                "-profile:v", "main",      # Changed from high to main
                "-level", "3.1",           # Changed from 4.0 to 3.1
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",     # Keep this for compatibility
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", # Keep ensuring even dimensions
                "-movflags", "+faststart", # Keep for web/streaming
            ])
        elif format_ext == '.wmv':
            cmd.extend([
                "-c:v", "wmv2",
                "-b:v", "2M",
                "-pix_fmt", "yuv420p",
            ])
        elif format_ext == '.avi':
            cmd.extend([
                "-c:v", "mpeg4",
                "-q:v", "4",
                "-pix_fmt", "yuv420p",
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ])

        cmd.append(output_name)

        print(f"Running FFmpeg command: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        print(f"FFmpeg encoding completed in {elapsed:.2f} seconds.")
        
        # Clean up temp files
        print("Cleaning up temporary files...")
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
        return True
    except Exception as e:
        print(f"FFmpeg processing error: {e}")
        return False

def make_video(image_folder, output_name, duration, reverse, skip_start, skip_end, durations, resize_width, resize_height, override, output_format="mp4"):
    # Ensure output filename has the correct extension
    output_name = ensure_correct_extension(output_name, output_format)
    
    # Check if the output file already exists
    if os.path.exists(output_name) and not override:
        while True:
            response = input(f"The file '{output_name}' already exists. Do you want to override it? (y/n): ").strip().lower()
            if response == 'y':
                break
            elif response == 'n':
                output_name = input("Please provide a new name for the output file or press Ctrl+C to cancel: ").strip()
                if not output_name:
                    print("Invalid file name. Please try again.")
                elif not os.path.exists(output_name):
                    break
            else:
                print("Invalid response. Please enter 'y' or 'n'.")

    # Get list of images in the folder - support more image formats
    supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.heic', '.heif')
    images = []
    for file in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file)
        if file.lower().endswith(supported_extensions) or is_valid_image(file_path):
            images.append(file)
    images.sort()
    
    # Apply reverse and skip logic
    if reverse:
        images = images[::-1]
    images = images[skip_start:len(images) - skip_end]

    if not images:
        print("No images found in the specified folder after applying skip logic.")
        return
        
    # Normalize durations to handle negative indices
    durations = normalize_durations(durations, len(images))

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = load_image(first_image_path)

    # Resize logic
    if resize_width:
        aspect_ratio = frame.shape[1] / frame.shape[0]
        resize_height = resize_height or int(resize_width / aspect_ratio)
        frame = cv2.resize(frame, (resize_width, resize_height))
    else:
        resize_width, resize_height = frame.shape[1], frame.shape[0]

    if output_format.lower() == "gif":
        frames = []
        print(f"Processing {len(images)} images for GIF creation...")
        
        for idx, image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = load_image(image_path)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if resize_width:
                frame_rgb = cv2.resize(frame_rgb, (resize_width, resize_height))
            
            frame_duration = next((d for i, d in durations if i == idx), duration)
            
            frame_count = int(frame_duration / duration)
            for _ in range(frame_count):
                frames.append(frame_rgb)
                
            print_progress(idx + 1, len(images), prefix='Processing GIF Frames:', suffix='Complete')
        
        print("\nSaving GIF...")
        imageio.mimsave(output_name, frames, duration=duration, loop=0)
        print(f"GIF saved as {output_name}")
    else:
        if check_ffmpeg():
            print("FFmpeg found. Using FFmpeg for video creation.")
            if make_video_ffmpeg(image_folder, output_name, duration, images, resize_width, resize_height, durations):
                print(f"Video saved as {output_name}")
                if output_format.lower() == 'mp4':
                    print("Attempted settings for improved Windows Media Player compatibility.")
                else:
                    print("Video created successfully with FFmpeg.")
                return
            else:
                print("FFmpeg video creation failed. Falling back to OpenCV (if possible).")

        print("Using OpenCV for video creation.")
        print("WARNING: OpenCV video creation may result in files incompatible with some players (like Windows Media Player).")
        print("For best results, please install FFmpeg.")

        format_lower = output_format.lower()
        codec_tag = OPENCV_FALLBACK_CODECS.get(format_lower)

        if not codec_tag:
            print(f"Error: Unsupported format '{output_format}' for OpenCV fallback.")
            print(f"Supported formats for OpenCV fallback: {', '.join(OPENCV_FALLBACK_CODECS.keys())}")
            if 'mp4' in OPENCV_FALLBACK_CODECS:
                print("Attempting MP4 format as a last resort.")
                format_lower = 'mp4'
                codec_tag = OPENCV_FALLBACK_CODECS['mp4']
                output_name = ensure_correct_extension(os.path.splitext(output_name)[0], 'mp4')
            else:
                print("Cannot proceed without a supported format.")
                return

        print(f"Using OpenCV with format: {format_lower}, codec: {codec_tag}")

        try:
            if resize_width % 2 != 0:
                print(f"Warning: Adjusting width from {resize_width} to {resize_width - 1} for codec compatibility.")
                resize_width -= 1
            if resize_height % 2 != 0:
                print(f"Warning: Adjusting height from {resize_height} to {resize_height - 1} for codec compatibility.")
                resize_height -= 1

            if resize_width <= 0 or resize_height <= 0:
                print(f"Error: Invalid dimensions after adjustment ({resize_width}x{resize_height}). Cannot create video.")
                return

            fourcc = cv2.VideoWriter_fourcc(*codec_tag)
            video = cv2.VideoWriter(output_name, fourcc, 1 / duration, (resize_width, resize_height))

            if not video.isOpened():
                print(f"Error: Could not open video writer with codec {codec_tag} for format {format_lower}.")
                print("This might be due to missing codecs on your system or an unsupported format/codec combination with OpenCV.")
                print("Please try installing FFmpeg for more reliable video creation.")
                return

            total_frames = 0
            for idx, _ in enumerate(images):
                frame_duration = next((d for i, d in durations if i == idx), duration)
                frame_count = int(frame_duration / duration)
                total_frames += frame_count
            
            print(f"Creating video with {total_frames} frames...")
            frames_written = 0
            
            for idx, image in enumerate(images):
                image_path = os.path.join(image_folder, image)
                frame = load_image(image_path)
                
                if frame is None:
                    print(f"\nWarning: Could not read image: {image_path}. Skipping this frame.")
                    continue

                if resize_width:
                    frame = cv2.resize(frame, (resize_width, resize_height))

                frame_duration = next((d for i, d in durations if i == idx), duration)
                frame_count = int(frame_duration / duration)

                for _ in range(frame_count):
                    video.write(frame)
                    frames_written += 1
                    if frames_written % 5 == 0 or frames_written == total_frames:
                        print_progress(frames_written, total_frames, prefix='Creating Video:', suffix='Complete')

            video.release()
            print(f"\nVideo saved as {output_name} using OpenCV.")
            print("Note: If you encounter playback issues, try using VLC Media Player or install FFmpeg and rerun the script.")
        except Exception as e:
            print(f"Error creating video with OpenCV: {e}")
            if 'fourcc' in locals() and not video.isOpened():
                print(f"Failed to initialize VideoWriter with FOURCC code: {codec_tag}. Check codec availability.")
            print("Consider installing FFmpeg for better video encoding capabilities.")

def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
            return True
    except Exception:
        return False

def load_image(file_path):
    if file_path.lower().endswith(('.heic', '.heif')):
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                open_cv_image = np.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                return open_cv_image
        except Exception as e:
            print(f"Error loading HEIC image {file_path}: {e}")
            return None
    else:
        return cv2.imread(file_path)

def ensure_correct_extension(filename, format_type):
    correct_extension = FORMAT_EXTENSIONS.get(format_type.lower(), ".mp4")
    base, ext = os.path.splitext(filename)
    if not ext or ext.lower() != correct_extension:
        return base + correct_extension
    return filename

def normalize_durations(duration_list, num_images):
    normalized = []
    if num_images == 0:
        return []
    for index, duration in duration_list:
        original_index = index
        if index < 0:
            index = num_images + index
        if 0 <= index < num_images:
            normalized.append((index, duration))
        else:
            print(f"Warning: Duration index {original_index} (resolved to {index}) is out of bounds for {num_images} images. Skipping.")
    return normalized

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a stop-motion video from images in a folder.")
    parser.add_argument("--folder", type=str, default="./img", help="Path to the folder containing images.")
    parser.add_argument("--format", type=str, default="mp4", 
                       choices=list(FORMAT_EXTENSIONS.keys()), 
                       help=f"Output format: {', '.join(FORMAT_EXTENSIONS.keys())}.")
    parser.add_argument("--output", type=str, default="output", help="Name of the output file (extension will be added based on format if missing).")
    parser.add_argument("--duration", type=float, default=0.2, help="Default duration of each frame in seconds.")
    parser.add_argument("--reverse", action="store_true", help="Reverse the order of images.")
    parser.add_argument("--skip_start", type=int, default=0, help="Number of images to skip at the start.")
    parser.add_argument("--skip_end", type=int, default=0, help="Number of images to skip at the end.")
    parser.add_argument("--di", type=str, default="[(-1,1)]", 
                       help="List of (index, duration_in_seconds) for specific frames. Negative indices like -1 (last image) are supported.")
    parser.add_argument("--resize_width", type=int, default=None, help="Resize width of the images (maintains aspect ratio if height is not set).")
    parser.add_argument("--resize_height", type=int, default=None, help="Resize height of the images.")
    parser.add_argument("--override", action="store_true", help="Override the output file if it already exists (default: False).")
    args = parser.parse_args()

    args.output = ensure_correct_extension(args.output, args.format)

    try:
        import ast
        durations = ast.literal_eval(args.di)
        if not isinstance(durations, list):
            raise ValueError("Input is not a list.")
        if not all(isinstance(item, (list, tuple)) and len(item) == 2 for item in durations):
            raise ValueError("List items are not all pairs (index, duration).")
        if not all(isinstance(item[0], int) and isinstance(item[1], (int, float)) for item in durations):
            raise ValueError("Index must be int, duration must be number.")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing --di argument: '{args.di}'. Expected a list of pairs like '[(0, 1.5), (-1, 2)]'.")
        print(f"Details: {e}")
        exit(1)

    make_video(args.folder, args.output, args.duration, args.reverse, args.skip_start, args.skip_end, durations, 
              args.resize_width, args.resize_height, args.override, args.format)