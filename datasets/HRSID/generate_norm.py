import numpy as np
import os
import cv2
import pdb

def apply_mask_to_image(image, mask):
        """
        Fills in the masked areas of the image by randomly picking pixels from the
        background (areas not covered by the mask) and assigning those values to
        the masked areas.

        Parameters:
            image (np.array): The input image with anomalies.
            mask (np.array): The mask indicating anomalous regions (non-zero values).

        Returns:
            np.array: The image with anomalies filled in.
        """

        # Create the output image by copying the original
        output_image = image.copy()

        # Get the indices of the background pixels (mask == 0)
        background_indices = np.where(mask == 0)

        # Get the background pixel values (pixels where the mask is 0)
        background_pixels = image[background_indices]

        # Check if there are any background pixels
        if len(background_pixels) == 0:
            return image

        # Iterate over the masked pixels (mask == 1)
        masked_indices = np.where(mask == 255)
        for idx in zip(*masked_indices):
            # Randomly select a background pixel
            random_pixel = background_pixels[np.random.randint(len(background_pixels))]

            # Replace the masked pixel with the selected background pixel
            output_image[idx] = random_pixel

        return output_image

def process_images_and_masks(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through images and masks
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust formats if needed
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            # Check if corresponding mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: No mask found for {filename}, skipping.")
                continue
            
            # Load image and mask
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error loading {filename} or its mask, skipping.")
                continue
            
            # Apply the mask processing function
            processed_image = apply_mask_to_image(image, mask)

            # Save the processed image
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, processed_image)
            print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    image_dir = './test/anom'  
    mask_dir = './ground_truth/anom'    
    output_dir = './test/norm' 

    process_images_and_masks(image_dir, mask_dir, output_dir)







