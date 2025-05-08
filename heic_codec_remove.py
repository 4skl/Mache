import os
from PIL import Image, ExifTags
import pillow_heif
import argparse

def convert_heic_to_format(input_path, output_format="png", output_dir=None):
    """
    Converts HEIC images to the specified format (png, jpeg, webp) while preserving metadata.

    :param input_path: Path to the HEIC file or folder containing HEIC files.
    :param output_format: Desired output format ("png", "jpeg", "webp").
    :param output_dir: Directory to save the converted images. Defaults to the input file's directory.
    """
    if output_format.lower() not in ["png", "jpeg", "webp"]:
        raise ValueError("Output format must be 'png', 'jpeg', or 'webp'.")

    if os.path.isdir(input_path):
        files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".heic")]
    elif input_path.lower().endswith(".heic"):
        files = [input_path]
    else:
        raise ValueError("Input path must be a HEIC file or a folder containing HEIC files.")

    for file in files:
        try:
            # Use pillow-heif to register HEIF opener for PIL
            pillow_heif.register_heif_opener()
            image = Image.open(file)

            # Preserve metadata (EXIF)
            exif_data = image.info.get("exif")

            # Determine output path
            out_dir = output_dir or os.path.dirname(file)
            output_file = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0] + f".{output_format}")

            # Ensure output directory exists
            os.makedirs(out_dir, exist_ok=True)

            # Save image with metadata if available
            save_kwargs = {"format": output_format.upper()}
            if exif_data and output_format.lower() in ["jpeg", "jpg"]:
                save_kwargs["exif"] = exif_data

            image.save(output_file, **save_kwargs)
            print(f"Converted: {file} -> {output_file}")
        except Exception as e:
            print(f"Failed to convert {file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HEIC images to PNG, JPEG, or WEBP while preserving metadata.")
    parser.add_argument("input_path", help="Path to the HEIC file or folder containing HEIC files.")
    parser.add_argument("output_format", choices=["png", "jpeg", "webp"], help="Desired output format (png, jpeg, webp).")
    parser.add_argument("--output_dir", help="Directory to save the converted images. Defaults to the input file's directory.", default=None)

    args = parser.parse_args()

    convert_heic_to_format(args.input_path, args.output_format, args.output_dir)