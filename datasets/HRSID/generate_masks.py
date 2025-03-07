import json
import os
import numpy as np
import cv2
from PIL import Image

def create_segmentation_masks(json_path, images_dir, output_dir):
    """Generate and save segmentation masks from COCO-format JSON annotations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping of image_id to file_name and dimensions
    image_info = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in data["images"]}
    
    # Process each annotation
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_info:
            continue
        
        file_name, width, height = image_info[image_id]
        mask_path = os.path.join(output_dir, file_name)
        
        # Initialize blank mask
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw segmentation mask
        segmentation = ann["segmentation"]
        for seg in segmentation:
            polygon = np.array(seg, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [polygon], color=255)
        
        # Save mask as PNG
        Image.fromarray(mask).save(mask_path)

if __name__ == "__main__":
    json_path = './annotations/test2017.json'  
    images_dir = "./test_images"  # Path to SAR images
    output_dir = "./test_masks"  # Directory to save PNG masks
    
    create_segmentation_masks(json_path, images_dir, output_dir)
