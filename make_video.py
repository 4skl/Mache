import os
import cv2
import argparse
import imageio  # Add imageio for GIF creation
from PIL import Image, ExifTags  # Replace imghdr with Pillow
from pathlib import Path
import numpy as np  # Import numpy for the codec detection
import pillow_heif  # Import pillow_heif for HEIC support
import subprocess  # For checking FFmpeg availability
import sys  # For progress bar output
import time  # For progress updates
from datetime import datetime  # For metadata date handling

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

def get_image_metadata_text(image_path, metadata_type="date", debug_mode=False):
    """Extracts specified metadata from an image file."""
    try:
        # For filename metadata type, just return the filename
        if metadata_type == "filename":
            return os.path.basename(image_path)
        
        # Always open with PIL for EXIF handling
        with Image.open(image_path) as img:
            # For date metadata types, specifically prioritize original capture date
            exif_data = {}
            
            # Try to get EXIF data - with additional error handling approaches
            try:
                # Method 1: Standard _getexif()
                exif_raw = img._getexif()
                if exif_raw:
                    for tag_id, value in exif_raw.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        exif_data[tag_id] = value
                        if debug_mode:
                            print(f"EXIF: {tag_name} ({tag_id}) = {value}")
            except (AttributeError, Exception) as e:
                if debug_mode:
                    print(f"Standard EXIF extraction failed: {e}")
                
                # Method 2: Try PIL's getexif() (newer method)
                try:
                    exif_raw = img.getexif()
                    if exif_raw:
                        for tag_id, value in exif_raw.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            exif_data[tag_id] = value
                            if debug_mode:
                                print(f"EXIF via getexif(): {tag_name} ({tag_id}) = {value}")
                except Exception as e:
                    if debug_mode:
                        print(f"PIL getexif() failed: {e}")

            # Check for EXIF in info dictionary for formats like PNG
            if hasattr(img, 'info') and 'exif' in img.info:
                if debug_mode:
                    print(f"Found EXIF in image info dictionary")

            # Define priority order for date tags with better comments
            date_tag_priority = [
                # Tag ID → Name mapping with highest priority first
                (36867, 'DateTimeOriginal'),    # When photo was taken (highest priority)
                (36868, 'DateTimeDigitized'),   # When photo was stored digitally
                (306, 'DateTime'),              # When file was modified in-camera
                (50971, 'PreviewDateTime'),     # Preview image date
                (40960, 'FlashpixVersion')      # May indicate date in some formats
            ]
            
            # For all metadata display (full metadata mode)
            if metadata_type == "all" or metadata_type == "full":
                metadata = []
                # Add filename
                metadata.append(f"File: {os.path.basename(image_path)}")
                
                # Try to get creation date with clear labeling of source
                date_found = False
                
                # Start with EXIF data - highest priority
                if exif_data:
                    if debug_mode:
                        print(f"Found {len(exif_data)} EXIF tags in {os.path.basename(image_path)}")
                    
                    # Look for date tags in priority order
                    for tag_id, tag_name in date_tag_priority:
                        if tag_id in exif_data and exif_data[tag_id] and str(exif_data[tag_id]).strip():
                            dt_str = str(exif_data[tag_id]).strip()
                            try:
                                # Try common EXIF date format
                                dt_obj = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                                metadata.append(f"{tag_name}: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}")
                                if debug_mode:
                                    print(f"Using EXIF {tag_name} date: {dt_str}")
                                date_found = True
                                break
                            except ValueError:
                                try:
                                    # Try alternate format
                                    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                                    metadata.append(f"{tag_name}: {dt_str}")
                                    date_found = True
                                    break
                                except ValueError:
                                    # Just add as raw string if can't parse
                                    metadata.append(f"{tag_name}: {dt_str}")
                                    date_found = True
                                    break
                
                # If no EXIF date found, try file system dates
                if not date_found:
                    # Try to get original file stats
                    try:
                        # Get file stats for creation time (platform-specific)
                        file_stats = os.stat(image_path)
                        
                        # Get creation time (Windows)
                        try:
                            creation_time = os.path.getctime(image_path)
                            ct_obj = datetime.fromtimestamp(creation_time)
                            metadata.append(f"File Created: {ct_obj.strftime('%Y-%m-%d %H:%M:%S')}")
                            if debug_mode:
                                print(f"Using file creation time: {ct_obj}")
                        except Exception as e:
                            if debug_mode:
                                print(f"Couldn't get file creation time: {e}")
                        
                        # Get modification time (all platforms)
                        file_time = os.path.getmtime(image_path)
                        dt_obj = datetime.fromtimestamp(file_time)
                        metadata.append(f"File Modified: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}")
                        if debug_mode:
                            print(f"Using file modification time: {dt_obj}")
                            
                    except Exception as e:
                        metadata.append(f"File dates unavailable: {str(e)}")
                        if debug_mode:
                            print(f"Error getting file dates: {e}")
                
                return "\n".join(metadata)
                
            # Handle date or datetime metadata types with proper priority
            if exif_data:
                # Check each tag in priority order
                for tag_id, tag_name in date_tag_priority:
                    if tag_id in exif_data and exif_data[tag_id] and str(exif_data[tag_id]).strip():
                        dt_str = str(exif_data[tag_id]).strip()
                        try:
                            # Standard EXIF date format
                            dt_obj = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
                            # Format based on requested type
                            if metadata_type == "date":
                                if debug_mode:
                                    print(f"Using EXIF {tag_name} date from {image_path}")
                                return dt_obj.strftime("%Y-%m-%d")
                            elif metadata_type == "datetime":
                                return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                return dt_str
                        except ValueError:
                            # Try alternate format or return as-is
                            try:
                                dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                                if metadata_type == "date":
                                    return dt_obj.strftime("%Y-%m-%d")
                                elif metadata_type == "datetime": 
                                    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                                else:
                                    return dt_str
                            except ValueError:
                                # If parsing fails, return the raw string with label
                                return f"{dt_str} (EXIF {tag_name})"
            
            if debug_mode:
                print(f"No usable EXIF date found in {image_path}, trying file timestamps")
        
            # Try file creation time before modification time
            try:
                # Get file creation time (works on Windows)
                creation_time = os.path.getctime(image_path)
                ct_obj = datetime.fromtimestamp(creation_time)
                if metadata_type == "date":
                    return f"{ct_obj.strftime('%Y-%m-%d')} (created)"
                elif metadata_type == "datetime":
                    return f"{ct_obj.strftime('%Y-%m-%d %H:%M:%S')} (created)"
                else:
                    return f"{ct_obj.strftime('%Y-%m-%d %H:%M:%S')} (file created)"
            except:
                if debug_mode:
                    print(f"Couldn't get file creation time, falling back to modification time")
                
            # Last resort: file modification time
            file_time = os.path.getmtime(image_path)
            dt_obj = datetime.fromtimestamp(file_time)
            if metadata_type == "date":
                return f"{dt_obj.strftime('%Y-%m-%d')} (mod)"
            elif metadata_type == "datetime":
                return f"{dt_obj.strftime('%Y-%m-%d %H:%M:%S')} (mod)"
            else:
                return f"{dt_obj.strftime('%Y-%m-%d %H:%M:%S')} (modified)"
                
    except Exception as e:
        print(f"Warning: Error getting metadata for {image_path}: {e}")
        return f"Error: {str(e)}"

def parse_metadata_color(color_str):
    """Parse color string to BGR tuple for OpenCV."""
    # Define common color names
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255)
    }
    
    # Check if it's a named color
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
        
    # Try to parse as a tuple
    try:
        # Check if it's in the format "(B,G,R)" or "B,G,R"
        if color_str.startswith("(") and color_str.endswith(")"):
            color_str = color_str[1:-1]
        
        components = [int(x.strip()) for x in color_str.split(",")]
        if len(components) == 3 and all(0 <= c <= 255 for c in components):
            return tuple(components)
    except:
        pass
        
    # Default to white if parsing fails
    print(f"Warning: Invalid color format '{color_str}'. Using white.")
    return color_map["white"]

def draw_metadata_on_frame(frame, text, position, color, font_scale, thickness):
    """Draw metadata text on the frame at the specified position."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate text size
    text_lines = text.split('\n')
    line_heights = []
    line_widths = []
    
    for line in text_lines:
        (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(text_height + baseline)
        line_widths.append(text_width)
    
    # Total text block height
    total_height = sum(line_heights)
    max_width = max(line_widths) if line_widths else 0
    
    # Determine position coordinates
    margin = 10
    if position == "top-left":
        x = margin
        y = margin + line_heights[0]
    elif position == "top-right":
        x = width - max_width - margin
        y = margin + line_heights[0]
    elif position == "bottom-left":
        x = margin
        y = height - total_height - margin + line_heights[0]
    else:  # bottom-right or default
        x = width - max_width - margin
        y = height - total_height - margin + line_heights[0]
    
    # Draw each line of text
    current_y = y
    for i, line in enumerate(text_lines):
        cv2.putText(frame, line, (x, current_y), font, font_scale, color, thickness, cv2.LINE_AA)
        if i < len(line_heights) - 1:  # Don't advance after the last line
            current_y += line_heights[i]

def make_video_ffmpeg(image_folder, output_name, duration, images, resize_width, resize_height, durations, 
                      show_metadata=False, metadata_type="date", metadata_position="bottom-right", 
                      metadata_color=(255,255,255), metadata_size=1.0, metadata_font_thickness=1,
                      metadata_debug=False):  # <-- add metadata_debug
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
            
            # Add metadata text if enabled
            if show_metadata:
                metadata_text = get_image_metadata_text(image_path, metadata_type, debug_mode=metadata_debug)  # <-- pass debug_mode
                if metadata_text:
                    draw_metadata_on_frame(frame, metadata_text, metadata_position, metadata_color, 
                                           metadata_size, metadata_font_thickness)
                
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

def make_video(image_folder, output_name, duration, reverse, skip_start, skip_end, durations, resize_width, resize_height, override, output_format="mp4",
               show_metadata=False, metadata_type="date", metadata_position="bottom-right", metadata_color="white", 
               metadata_size=1.0, metadata_font_thickness=1, metadata_debug=False):
    # Ensure output filename has the correct extension
    output_name = ensure_correct_extension(output_name, output_format)
    metadata_color_bgr = parse_metadata_color(metadata_color)

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
            
            # Add metadata text if enabled
            if show_metadata:
                metadata_text = get_image_metadata_text(image_path, metadata_type, debug_mode=metadata_debug)  # <-- pass debug_mode
                if metadata_text:
                    draw_metadata_on_frame(frame_rgb, metadata_text, metadata_position, metadata_color_bgr, 
                                           metadata_size, metadata_font_thickness)
            
            frame_duration = next((d for i, d in durations if i == idx), duration)
            
            frame_count = int(frame_duration / duration)
            for _ in range(frame_count):
                frames.append(frame_rgb)
                
            print_progress(idx + 1, len(images), prefix='Processing GIF Frames:', suffix='Complete')
        
        print("\nSaving GIF...")
        imageio.mimsave(output_name, frames, duration=duration, loop=0)
        print(f"GIF saved as {output_name}")
    else:
        # Always create AVI first for compatibility
        temp_avi = os.path.splitext(output_name)[0] + "_temp.avi"
        format_lower = "avi"
        codec_tag = OPENCV_FALLBACK_CODECS["avi"]

        print(f"Creating temporary AVI video: {temp_avi}")

        video = None
        try:
            start_time = time.time()
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
            video = cv2.VideoWriter(temp_avi, fourcc, 1 / duration, (adjusted_w, adjusted_h))
            if not video.isOpened():
                print(f"Error: Could not open video writer with codec {codec_tag} for AVI.")
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
                if frame.shape[1] != adjusted_w or frame.shape[0] != adjusted_h:
                    frame = cv2.resize(frame, (adjusted_w, adjusted_h))
                if show_metadata:
                    metadata_text = get_image_metadata_text(image_path, metadata_type, debug_mode=metadata_debug)
                    if metadata_text:
                        draw_metadata_on_frame(frame, metadata_text, metadata_position, metadata_color_bgr,
                                              metadata_size, metadata_font_thickness)
                if frame.ndim == 2:
                    frame = np.stack([frame]*3, axis=-1)
                elif frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                elif frame.shape[2] < 3:
                    print(f"Warning: Frame has less than 3 channels, skipping: {image_path}")
                    continue
                if frame.shape != (adjusted_h, adjusted_w, 3):
                    print(f"Error: Frame shape after processing is {frame.shape}, expected ({adjusted_h}, {adjusted_w}, 3). Skipping frame.")
                    continue
                frame = np.ascontiguousarray(frame)
                frame_duration = next((d for i, d in durations if i == idx), duration)
                frame_count = int(frame_duration / duration)
                for _ in range(frame_count):
                    video.write(frame)
                    frames_written += 1
                    if frames_written % 5 == 0 or frames_written == total_frames:
                        print_progress(frames_written, total_frames, prefix='Creating Video:', suffix='Complete')
            elapsed = time.time() - start_time
            print(f"\nTemporary AVI video saved as {temp_avi}.")
            print(f"Total processing time: {elapsed:.2f} seconds.")
        except Exception as e:
            print(f"Error creating video with OpenCV: {e}")
        finally:
            if video is not None and video.isOpened():
                video.release()
                print("OpenCV VideoWriter released.")

        # If requested format is avi, just rename/move the temp file
        if output_format.lower() == "avi":
            if temp_avi != output_name:
                os.replace(temp_avi, output_name)
            print(f"Video saved as {output_name} (AVI, maximum compatibility).")
            return

        # If FFmpeg is available, convert AVI to requested format
        if check_ffmpeg():
            print(f"FFmpeg found. Converting AVI to {output_format}...")
            cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_avi,
            ]
            format_ext = os.path.splitext(output_name)[1].lower()
            if format_ext == '.mp4':
                cmd += [
                    "-c:v", "libx264",
                    "-profile:v", "baseline",
                    "-level", "3.0",
                    "-b:v", "2M",
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-movflags", "+faststart",
                ]
            elif format_ext == '.wmv':
                cmd += [
                    "-c:v", "wmv2",
                    "-b:v", "2M",
                    "-pix_fmt", "yuv420p",
                ]
            elif format_ext == '.webm':
                cmd += [
                    "-c:v", "libvpx-vp9",
                    "-b:v", "2M",
                ]
            elif format_ext == '.mov':
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                ]
            elif format_ext == '.mkv':
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                ]
            else:
                cmd += [
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                ]
            cmd.append(output_name)
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                print(f"Leaving temporary AVI at {temp_avi}")
                return
            else:
                print(f"Video saved as {output_name} (converted from AVI).")
                os.remove(temp_avi)
        else:
            print("FFmpeg not found. Only AVI output is available.")
            print(f"Your video is at {temp_avi}. You may convert it manually if needed.")

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
    
    # Metadata display options
    parser.add_argument("--show_metadata", action="store_true", help="Show metadata on the video frames.")
    parser.add_argument("--metadata_type", type=str, default="date", 
                        choices=["date", "datetime", "filename", "all", "full"],
                        help="Type of metadata to display: date, datetime, filename, or all/full (default: date)")
    parser.add_argument("--metadata_position", type=str, default="bottom-right",
                        choices=["bottom-right", "bottom-left", "top-right", "top-left"],
                        help="Position of metadata text (default: bottom-right)")
    parser.add_argument("--metadata_color", type=str, default="white", 
                        help="Color of metadata text. Can be name (white, black, red, green, blue, yellow, cyan, magenta) or BGR values like '255,255,255'")
    parser.add_argument("--metadata_size", type=float, default=1.0,
                        help="Font size for metadata text (default: 1.0)")
    parser.add_argument("--metadata_font_thickness", type=int, default=1,
                        help="Font thickness for metadata text (default: 1)")
    parser.add_argument("--metadata_debug", action="store_true",
                        help="Enable debug output for metadata extraction")
    
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
              args.resize_width, args.resize_height, args.override, args.format,
              args.show_metadata, args.metadata_type, args.metadata_position,
              args.metadata_color, args.metadata_size, args.metadata_font_thickness, 
              metadata_debug=args.metadata_debug)

    overall_elapsed_time = time.time() - overall_start_time
    print(f"Total script execution time: {overall_elapsed_time:.2f} seconds.")