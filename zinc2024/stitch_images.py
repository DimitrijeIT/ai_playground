from PIL import Image
import os

def convert_and_stitch(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            original_image_path = os.path.join(folder_path, filename)
            original_image = Image.open(original_image_path)
            grayscale_image = original_image.convert("L")

            width, height = original_image.size
            stitched_image = Image.new('RGB', (2 * width, height))
            stitched_image.paste(original_image, (0, 0))
            stitched_image.paste(grayscale_image.convert("RGB"), (width, 0))

            output_image_path = os.path.join(output_folder, f"{filename}")
            stitched_image.save(output_image_path)

input_folder = "dataset_small/train"
output_folder = "dataset_stitch_small/train"
convert_and_stitch(input_folder, output_folder)
