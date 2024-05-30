import os
from PIL import Image
import pyheif


def convert_heic_to_jpg(heic_file_path, jpg_file_path):
    # Open HEIC image using pyheif
    heif_file = pyheif.read(heic_file_path)

    # Convert the HEIF image to a format compatible with PIL
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Calculate new size (half of the original size)
    new_size = (image.width // 2, image.height // 2)

    # Resize the image to half of its original size
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Save the image as a JPEG
    image.save(jpg_file_path, format="JPEG")


def batch_convert_heic_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.heic'):
            heic_file_path = os.path.join(directory, filename)
            jpg_file_path = os.path.splitext(heic_file_path)[0] + '.jpg'
            convert_heic_to_jpg(heic_file_path, jpg_file_path)
            print(f"Converted {heic_file_path} to {jpg_file_path}")


# Example usage
directory = '/home/yehezkely/Desktop/gs-new/let'
batch_convert_heic_to_jpg(directory)
