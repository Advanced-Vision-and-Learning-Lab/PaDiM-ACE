from absl import logging
from absl import flags
from absl import app

from multiprocessing import Pool
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

import json
import glob
import os
import pdb

from scipy.ndimage import gaussian_filter
import cv2
import mstar

flags.DEFINE_string('image_root', default='Datasets/Dataset', help='')
flags.DEFINE_string('dataset', default='soc', help='')
flags.DEFINE_boolean('is_train', default=False, help='')
flags.DEFINE_integer('chip_size', default=100, help='')
flags.DEFINE_integer('patch_size', default=94, help='')
flags.DEFINE_boolean('use_phase', default=False, help='')

FLAGS = flags.FLAGS

project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'PaDiM-LACE')
print(project_root)


def generate_mask(image, n_clusters=2, sigma=1, kernel_size=50):
    """
    Generate a filled mask for the image using KMeans clustering,
    with Gaussian blurring and morphological operations to fill shapes.
    
    Parameters:
        image (np.array): The input image as a NumPy array.
        n_clusters (int): The number of clusters for KMeans.
        sigma (float): Standard deviation for Gaussian blur. Higher values increase blurring.
        
    Returns:
        np.array: The filled mask with cluster labels for each pixel.
    """
    # Apply Gaussian blur to reduce noise and speckling
    smoothed_image = gaussian_filter(image, sigma=sigma)

    # Flatten the smoothed image for KMeans clustering
    reshaped_image = smoothed_image.reshape(-1, 1)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reshaped_image)
    labels = kmeans.labels_

    # Reshape the result back to the original image shape
    mask = labels.reshape(image.shape)

    # Convert the mask to uint8 format for OpenCV
    mask_uint8 = (mask * (255 // (n_clusters - 1))).astype(np.uint8)

    # Apply morphological operations to fill in the shapes
    circular_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filled_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, circular_kernel)

    num_labels, labels_im = cv2.connectedComponents(filled_mask.astype(np.uint8))

    # Find the largest component by size
    largest_label = 1  # Start from 1 to skip the background (0)
    largest_size = 0
    
    for label in range(1, num_labels):
        component_size = np.sum(labels_im == label)
        if component_size > largest_size:
            largest_size = component_size
            largest_label = label

    # Create a mask that keeps only the largest component
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels_im == largest_label] = 255  # Keep the largest component

    return cleaned_mask

def apply_mask_to_image(image, mask):
    return image


def data_scaling(chip):
    r = chip.max() - chip.min()
    return (chip - chip.min()) / r


def log_scale(chip):
    return np.log10(np.abs(chip) + 1)


def generate(src_path, dst_path, is_train, chip_size, patch_size, use_phase, dataset):
    if not os.path.exists(src_path):
        print(f'{src_path} does not exist')
        return
    if not os.path.exists(dst_path):
        os.makedirs(dst_path, exist_ok=True)

    anomalistic_dir = os.path.join(dst_path, 'anomalistic_images')
    normal_dir = os.path.join(dst_path, 'normal_images')
    mask_dir = os.path.join(dst_path, 'masks')
    json_dir = os.path.join(dst_path, 'json')

    for directory in [anomalistic_dir, normal_dir, mask_dir, json_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f'Target Name: {os.path.basename(dst_path)}')
    _mstar = mstar.MSTAR(
        name=dataset, is_train=is_train, chip_size=chip_size, patch_size=patch_size, use_phase=use_phase, stride=1
    )

    image_list = glob.glob(os.path.join(src_path, '*'))

    for path in image_list:
        label, _images = _mstar.read(path)
        for i, _image in enumerate(_images):
            name = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(json_dir, f'{name}-{i}.json'), mode='w', encoding='utf-8') as f:
                json.dump(label, f, ensure_ascii=False, indent=2)

            np.save(os.path.join(normal_dir, f'{name}-{i}.npy'), _image)  # Save normal images
            
            # Generate and save the mask
            mask = generate_mask(_image)
            np.save(os.path.join(mask_dir, f'{name}-{i}.npy'), mask)  # Save masks
            
            # Remove anomalies using blending based on the mask
            normal_image = apply_mask_to_image(_image, mask)  # Use blending
            np.save(os.path.join(normal_dir, f'{name}-{i}.npy'), normal_image)  # Save the modified image

def main(_):
    dataset_root = os.path.join(project_root, FLAGS.image_root, FLAGS.dataset)
    print(dataset_root)
    raw_root = os.path.join(dataset_root, 'raw')
    print(raw_root)

    mode = 'train' if FLAGS.is_train else 'test'

    output_root = os.path.join(dataset_root, mode)
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    arguments = [
        (
            os.path.join(raw_root, mode, target),
            os.path.join(output_root, target),
            FLAGS.is_train, FLAGS.chip_size, FLAGS.patch_size, FLAGS.use_phase, FLAGS.dataset
        ) for target in mstar.target_name[FLAGS.dataset]
    ]

    # with Pool(10) as p:
    #     p.starmap(generate, arguments)
    for target in mstar.target_name[FLAGS.dataset]:
        generate(os.path.join(raw_root, mode, target),
                os.path.join(output_root, target),
                FLAGS.is_train, 
                FLAGS.chip_size, 
                FLAGS.patch_size, 
                FLAGS.use_phase, 
                FLAGS.dataset)
        # pdb.set_trace()

if __name__ == '__main__':
    app.run(main)
