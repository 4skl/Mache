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
    "avi": "MJPG",  # Use MJPG for best AVI compatibility
    "mov": "mp4v",
    "wmv": "WMV2"
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
        frame_idx = 0  # Use a running frame index for sequential naming
        for idx, image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = load_image(image_path)
            
            if frame is None:
                print(f"\nWarning: Could not read image: {image_path}. Skipping this frame.")
                continue
                
            # Resize frame if needed
            if resize_width and resize_height:
                frame = cv2.resize(frame, (resize_width, resize_height))
                
            # Save frame(s) to temp directory for duration
            frame_duration = next((d for i, d in durations if i == idx), duration)
            frame_count = int(frame_duration / duration)
            for _ in range(frame_count):
                frame_path = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
                cv2.imwrite(frame_path, frame)
                frame_idx += 1
                processed_frames += 1
                print_progress(processed_frames, total_frames, prefix='Preparing Frames:', suffix='Complete')
        
        print("\nEncoding video with FFmpeg...")
        
        fps = 1 / duration
        format_ext = os.path.splitext(output_name)[1].lower()
        
        # Use sequential frame pattern for Windows compatibility
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", os.path.join(temp_dir, "frame_%05d.png"),
        ]

        # Format-specific codec and options
        if format_ext == '.mp4':
            print("Using FFmpeg with Baseline Profile Level 3.0 for MP4 (maximum WMP compatibility)")
            cmd.extend([
                "-c:v", "libx264",
                "-profile:v", "baseline",
                "-level", "3.0",
                "-b:v", "2M",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-movflags", "+faststart",
            ])
        elif format_ext == '.wmv':
            cmd.extend([
                "-c:v", "wmv2",
                "-b:v", "2M",
                "-pix_fmt", "yuv420p",
            ])
        elif format_ext == '.avi':
            print("Using FFmpeg with MJPEG codec for AVI (maximum compatibility)")
            cmd.extend([
                "-c:v", "mjpeg",
                "-q:v", "3",
                "-pix_fmt", "yuvj422p",
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

    # Determine video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    first_frame_for_dims = load_image(first_image_path)
    if first_frame_for_dims is None:
        print(f"Error: Could not load the first image: {first_image_path}. Cannot determine dimensions.")
        return

    original_frame_width = first_frame_for_dims.shape[1]
    original_frame_height = first_frame_for_dims.shape[0]

    video_width = resize_width  # User-provided or None
    video_height = resize_height # User-provided or None

    if video_width and not video_height:
        # Width provided, calculate height
        aspect_ratio = original_frame_height / original_frame_width # height/width
        video_height = int(video_width * aspect_ratio)
    elif not video_width and video_height:
        # Height provided, calculate width
        aspect_ratio = original_frame_width / original_frame_height # width/height
        video_width = int(video_height * aspect_ratio)
    elif not video_width and not video_height:
        # Neither provided, use original dimensions of the first frame
        video_width = original_frame_width
        video_height = original_frame_height
    # If both video_width and video_height are provided by user, they are used directly.

    print(f"Target video dimensions: {video_width}x{video_height}")

    if output_format.lower() == "gif":
        frames = []
        print(f"Processing {len(images)} images for GIF creation...")
        
        for idx, image in enumerate(images):
            image_path = os.path.join(image_folder, image)
            frame = load_image(image_path)
            if frame is None:
                print(f"\nWarning: Could not read image: {image_path}. Skipping this frame.")
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use calculated video_width and video_height for resizing
            if video_width and video_height:
                frame_rgb = cv2.resize(frame_rgb, (video_width, video_height))
            
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
            # Pass calculated video_width and video_height to FFmpeg function
            if make_video_ffmpeg(image_folder, output_name, duration, images, video_width, video_height, durations):
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
        print("TIP: If you must use OpenCV, try '--format avi' for best compatibility, and play the result with VLC Media Player.")
        print("NOTE: If your video does not play in Windows Media Player or Movies & TV, this is a limitation of OpenCV's video writer. The only reliable way to produce universally compatible video files is to use FFmpeg with H.264 or MPEG-4 codecs. VLC Media Player can play almost any video produced by OpenCV.")

        format_lower = output_format.lower()
        codec_tag = OPENCV_FALLBACK_CODECS.get(format_lower)

        if not codec_tag:
            print(f"Error: Unsupported format '{output_format}' for OpenCV fallback.")
            print(f"Supported formats for OpenCV fallback: {', '.join(OPENCV_FALLBACK_CODECS.keys())}")
            if 'avi' in OPENCV_FALLBACK_CODECS:
                print("Attempting AVI format as a last resort.")
                format_lower = 'avi'
                codec_tag = OPENCV_FALLBACK_CODECS['avi']
                output_name = ensure_correct_extension(os.path.splitext(output_name)[0], 'avi')
            else:
                print("Cannot proceed without a supported format.")
                return

        print(f"Using OpenCV with format: {format_lower}, codec: {codec_tag}")

        video = None  # Initialize video object outside try block
        try:
            start_time = time.time()  # Start timing for OpenCV part

            # Use video_width and video_height for adjustments
            adjusted_w = video_width
            adjusted_h = video_height

            if adjusted_w % 2 != 0:
                print(f"Warning: Adjusting width from {adjusted_w} to {adjusted_w - 1} for codec compatibility.")
                adjusted_w -= 1
            if adjusted_h % 2 != 0:
                print(f"Warning: Adjusting height from {adjusted_h} to {adjusted_h - 1} for codec compatibility.")
                adjusted_h -= 1

            if adjusted_w <= 0 or adjusted_h <= 0:
                print(f"Error: Invalid dimensions after adjustment ({adjusted_w}x{adjusted_h}). Cannot create video.")
                return
            
            print(f"OpenCV VideoWriter dimensions: {adjusted_w}x{adjusted_h}")

            fourcc = cv2.VideoWriter_fourcc(*codec_tag)
            video = cv2.VideoWriter(output_name, fourcc, 1 / duration, (adjusted_w, adjusted_h))

            if not video.isOpened():
                print(f"Error: Could not open video writer with codec {codec_tag} for format {format_lower}.")
                print("This might be due to missing codecs on your system or an unsupported format/codec combination with OpenCV.")
                print("Please try installing FFmpeg for more reliable video creation.")
                print("TIP: If you cannot install FFmpeg, try using '--format avi' for best compatibility with Windows Media Player, and play the result with VLC Media Player.")
                print("NOTE: If you see a 'Permission denied' error, make sure the output file is not open in any program and you have write permissions in the folder.")
                print("NOTE: If you see an OpenCV error about 'cv::VideoWriter_Images', check that your output filename ends with .avi or .mp4 and is not a folder or pattern.")
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

                # Resize frame first to adjusted dimensions
                if frame.shape[1] != adjusted_w or frame.shape[0] != adjusted_h:
                    frame = cv2.resize(frame, (adjusted_w, adjusted_h))

                # Ensure frame has exactly 3 channels (BGR)
                if frame.ndim == 2:
                    # Grayscale: stack to 3 channels
                    frame = np.stack([frame]*3, axis=-1)
                elif frame.shape[2] > 3:
                    # More than 3 channels: crop to first 3
                    frame = frame[:, :, :3]
                elif frame.shape[2] < 3:
                    print(f"Warning: Frame has less than 3 channels, skipping: {image_path}")
                    continue

                # Final check: shape must be (adjusted_h, adjusted_w, 3)
                if frame.shape != (adjusted_h, adjusted_w, 3):
                    print(f"Error: Frame shape after processing is {frame.shape}, expected ({adjusted_h}, {adjusted_w}, 3). Skipping frame.")
                    continue
                
                # Ensure frame is C-contiguous
                frame = np.ascontiguousarray(frame)

                frame_duration = next((d for i, d in durations if i == idx), duration)
                frame_count = int(frame_duration / duration)

                for _ in range(frame_count):
                    video.write(frame)
                    frames_written += 1
                    if frames_written % 5 == 0 or frames_written == total_frames:
                        print_progress(frames_written, total_frames, prefix='Creating Video:', suffix='Complete')

            elapsed = time.time() - start_time  # End timing
            print(f"\nVideo saved as {output_name} using OpenCV.")
            print(f"Total processing time: {elapsed:.2f} seconds.")
            print("Note: If you encounter playback issues, try using VLC Media Player or install FFmpeg and rerun the script.")
        except Exception as e:
            print(f"Error creating video with OpenCV: {e}")
            if 'fourcc' in locals() and video is not None and not video.isOpened(): # Check if video was initialized
                print(f"Failed to initialize VideoWriter with FOURCC code: {codec_tag}. Check codec availability.")
            print("Consider installing FFmpeg for better video encoding capabilities.")
        finally:
            if video is not None and video.isOpened():
                video.release()
                print("OpenCV VideoWriter released.")

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
    parser.add_argument("--format", type=str, default="avi", 
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
    
    overall_start_time = time.time()

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

    overall_elapsed_time = time.time() - overall_start_time
    print(f"Total script execution time: {overall_elapsed_time:.2f} seconds.")