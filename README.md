# Video/GIF Creator from Images

This project is a simple tool to create videos or GIFs from a collection of images stored in the `/img` directory.

!!! Still under development !!!

## Features
- Convert images into a video.
- Generate GIFs from images.
- Easy-to-use and lightweight.
- Supports multiple formats: mp4, gif, webm, avi, mov, mkv, wmv.
- Custom frame durations and skipping images.
- HEIC/HEIF image support.

## Requirements
- Python 3.x
- Required libraries: `opencv-python`, `Pillow`, `imageio`, `pillow-heif`, `numpy`
- (Optional but recommended) [FFmpeg](https://ffmpeg.org/) for best video compatibility

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Place your images in the `/img` directory.
2. Run the script:
    ```bash
    python make_video.py --folder ./img --format mp4 --output myvideo --duration 0.2 --override
    ```
   - See all options with:
    ```bash
    python make_video.py --help
    ```
3. Example: Custom durations for first and last frame, resize to 720p height:
    ```bash
    python make_video.py --resize_height 720 --format mp4 --di "[(0,1),(-1,2)]" --override
    ```

## Directory Structure
```
/img          # Directory containing input images
make_video.py # Main script to run the project
README.md     # Project documentation
requirements.txt
```

## License
This project is licensed under the WTFPL License.
