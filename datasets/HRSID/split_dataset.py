import os
import shutil
import json

def split_dataset(json_path, images_dir, train_dir, test_dir):
    """Separate images into train and test directories based on COCO JSON."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Identify train and test images
    for img in data["images"]:
        file_name = img["file_name"]
        src_path = os.path.join(images_dir, file_name)
        
        # Check if the file exists before moving
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found.")
            continue
        
        # Determine if the image belongs to train or test set
        if "train" in json_path:
            dest_path = os.path.join(train_dir, file_name)
        else:
            dest_path = os.path.join(test_dir, file_name)
        
        shutil.copy2(src_path, dest_path)
        
    print("Dataset split completed.")

if __name__ == "__main__":
    json_path = './annotations/test2017.json'  
    images_dir = './images'  # Path to SAR images
    train_dir = './train_images'  # Directory to store training images
    test_dir = './test_images'  # Directory to store test images
    
    split_dataset(json_path, images_dir, train_dir, test_dir)
