python make_video.py --resize_height 720 --di "[(0,1),(-1,2)]" --override --show_metadata
ffmpeg -i output.avi -c:v h264 -crf 0 -c:a flac output.mp4