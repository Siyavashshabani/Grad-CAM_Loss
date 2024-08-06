import os 
import numpy as np
from skimage import io, color, measure, segmentation, morphology
from scipy import ndimage as ndi
import cv2
import czifile
import os 
import numpy as np
import matplotlib.pyplot as plt


def dilate_mask(mask, size):
    selem = morphology.disk(size)
    dilated_mask = morphology.dilation(mask, selem)
    return dilated_mask

def process_mask(mask_path, image):

    # print(image.shape)

    # Load the label image
    msk = io.imread(mask_path)
    print("start after read shape", msk.shape)
    # Ensure the image is 2D if it has a color channel
    image = image[:, :msk.shape[0], :msk.shape[1],: ]
    if msk.ndim == 3:
        msk = color.rgb2gray(msk)

    # Apply a binary threshold to the image
    binary_img = msk > 0

    # Compute the distance transform
    distance = ndi.distance_transform_edt(binary_img)

    # Find local maxima in the distance transform
    local_maxi = morphology.local_maxima(distance)

    # Label the local maxima as markers for the watershed
    markers = measure.label(local_maxi)

    # Apply watershed segmentation
    label2 = segmentation.watershed(-distance, markers, mask=binary_img)
    labels = morphology.area_opening(label2, 130)

    # Get the number of unique objects after watershed
    num_objects = labels.max()
    print("Number of objects:", num_objects)

    # Process each object to create inner part masks
    props = measure.regionprops(labels)
    all_masks = np.zeros((len(props), msk.shape[0], msk.shape[1]), dtype=np.uint8)

    for nid, prop in enumerate(props):
        output_image = np.zeros_like(msk, dtype=np.uint8)
        region_coords = prop.coords
        output_image[region_coords[:, 0], region_coords[:, 1]] = 1

        # Define the erosion kernel size
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Perform erosion
        eroded_image = cv2.erode(output_image, kernel, iterations=1)
        dapi_dilated = dilate_mask(output_image, 5)

        # Create a mask to exclude the inner part
        inner_part_mask = dapi_dilated - eroded_image

        # Store the inner_part_mask in the array
        all_masks[nid] = inner_part_mask

    return all_masks, image


def calculate_bounding_boxes(objects_array, padding=50):
    bounding_boxes = []
    height, width = objects_array.shape[1], objects_array.shape[2]  # Assuming shape [num_objects, H, W]
    
    for i in range(objects_array.shape[0]):
        object_slice = objects_array[i, :, :]
        
        # Find non-zero points (where object exists)
        coords = np.column_stack(np.where(object_slice > 0))
        
        if coords.size == 0:
            continue
        
        # Determine bounding box coordinates with padding
        x_min, y_min = coords.min(axis=0) - padding
        x_max, y_max = coords.max(axis=0) + padding
        
        # Ensure the coordinates do not go out of image boundaries
        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, width)
        y_max = min(y_max, height)
        
        # Append coordinates as (x_min, y_min, width, height)
        bounding_boxes.append([y_min, x_min, y_max - y_min, x_max - x_min])
        
    return bounding_boxes

def extract_patches(image, mask, bounding_boxes): 
    image_patches = []
    mask_patches = []
    
    for i, box in enumerate(bounding_boxes):
        # print("box:", box)
        y_min, x_min, width, height = box
        
        # Check if width or height are negative
        if width < 0 or height < 0:
            print(f"Skipping box with invalid dimensions: {box}")
            continue
        
        # Extract patches from both image and mask using the same bounding box
        # image=image.T
        image_patch = image[:, x_min:x_min + height, y_min:y_min + width]
        mask_patch = mask[i, x_min:x_min + height, y_min:y_min + width]

        print("image_patch.shape:", image_patch.shape)
        print("mask_patch.shape:", mask_patch.shape)        

        image_patches.append(image_patch)
        mask_patches.append(mask_patch)
    
    return image_patches, mask_patches