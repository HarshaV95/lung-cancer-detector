import numpy as np
import cv2
from PIL import Image
import os

def generate_ct_scan_like_image():
    """Generate a synthetic image that mimics CT scan characteristics."""
    # Create a 512x512 image (common CT scan resolution)
    img = np.zeros((512, 512), dtype=np.uint8)
    
    # Add a circular structure (like a chest cavity)
    cv2.circle(img, (256, 256), 200, 255, -1)
    
    # Add some internal structures
    cv2.ellipse(img, (256, 256), (100, 150), 0, 0, 360, 200, -1)
    
    # Add some noise and texture
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Apply Gaussian blur to smooth
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert to RGB (3 channels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img_rgb

def generate_normal_image():
    """Generate a non-CT scan test image."""
    # Create a colorful gradient image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Add some colorful patterns
    for i in range(512):
        for j in range(512):
            img[i, j] = [(i + j) % 256, i % 256, j % 256]
    
    return img

def save_test_images():
    """Generate and save test images."""
    # Create test images directory
    os.makedirs("test_images", exist_ok=True)
    
    # Generate and save CT scan-like image
    ct_image = generate_ct_scan_like_image()
    ct_image_pil = Image.fromarray(ct_image)
    ct_image_pil.save("test_images/synthetic_ct_scan.png")
    
    # Generate and save normal image
    normal_image = generate_normal_image()
    normal_image_pil = Image.fromarray(normal_image)
    normal_image_pil.save("test_images/non_ct_image.png")

if __name__ == "__main__":
    save_test_images()
