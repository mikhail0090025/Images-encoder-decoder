import kagglehub
import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import main_variables as mv

np.random.seed(42)

# Download latest version
path = kagglehub.dataset_download("prasunroy/natural-images")

print("Path to dataset files:", path)

def image_to_numpy(path_to_image):
    try:
        img = Image.open(path_to_image)
        img = img.convert("RGB")
        img_resized = img.resize(mv.start_resolution, Image.Resampling.LANCZOS)
        img_array = (np.array(img_resized) / 127.5).astype(np.float32)
        img_array = img_array - 1
        return img_array
    except UnidentifiedImageError:
        print(f"Error loading {path}: not an image, skipping")

def get_images():
    if os.path.exists('images.npz'):
        print("Save was found. Loading...")
        images = np.load('images.npz')['images']
        print(f"Images shape: {images.shape}")
        return images
    
    print("Save was not found. Creating...")
    images = []
    all_dirs = os.listdir(os.path.join(path, 'natural_images'))
    for directory in all_dirs:
        print(f"Path: {os.path.join(path, 'natural_images', directory)}")
        max_files_for_folder = 200
        for image_path in os.listdir(os.path.join(path, 'natural_images', directory))[:max_files_for_folder]:
            img = image_to_numpy(os.path.join(path, 'natural_images', directory, image_path))
            images.append(img)
        
    images = np.array(images)
    print(images)
    print(f"Images shape: {images.shape}")
    np.savez_compressed('images.npz', images=images)
    return images