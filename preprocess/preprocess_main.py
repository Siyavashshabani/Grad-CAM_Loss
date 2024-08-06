import os
print("first----------",os.getcwd())
import sys
sys.path.append('/home/sshabani/projects/Grad_CAMO')

print("second----------",os.getcwd())

import numpy as np
import czifile
from skimage import io, color
import argparse
from preprocess.utils import dilate_mask, process_mask, calculate_bounding_boxes

def save_patches(image, mask_channels, bounding_boxes, save_directory, file_name):
    os.makedirs(os.path.join(save_directory, 'image'), exist_ok=True)
    os.makedirs(os.path.join(save_directory, 'mask'), exist_ok=True)

    for i, box in enumerate(bounding_boxes):
        y_min, x_min, width, height = box
        if width < 0 or height < 0:
            print(f"Skipping box with invalid dimensions: {box}")
            continue
        
        image_patch = image[:, x_min:x_min + height, y_min:y_min + width, 0]
        mask_patch = mask_channels[i, x_min:x_min + height, y_min:y_min + width]

        image_patch_path = os.path.join(save_directory, 'image', f'{file_name}_image_patch_{i}.npy')
        mask_patch_path = os.path.join(save_directory, 'mask', f'{file_name}_mask_patch_{i}.npy')
        np.save(image_patch_path, image_patch)
        np.save(mask_patch_path, mask_patch)

def main(args):
    file_name = os.path.splitext(os.path.basename(args.image_path))[0]
    image = czifile.imread(args.image_path)
    mask_channels = process_mask(args.mask_path)
    bounding_boxes = calculate_bounding_boxes(mask_channels, padding=10)
    save_patches(image, mask_channels, bounding_boxes, args.save_directory, file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and masks.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("mask_path", type=str, help="Path to the mask file.")
    parser.add_argument("save_directory", type=str, help="Directory to save the patches.")

    args = parser.parse_args()
    main(args)


# python3 ./preprocess/preprocess_main.py /home/sshabani/projects/Grad_CAMO/data/lymphos/A1818_P0025_4MGRTumor_2.czi 
# /home/sshabani/projects/Grad_CAMO/data/lymphos/A1818_P0025_4MGRTumor_2_mask.png  /home/sshabani/projects/Grad_CAMO/data/preprocessed/lymph