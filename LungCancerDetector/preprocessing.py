import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import TARGET_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
import logging

logger = logging.getLogger(__name__)

def check_ct_characteristics(image):
    """
    Check if the image has characteristics typical of CT scans.

    Args:
        image: numpy array of image

    Returns:
        tuple: (bool, str) - (is_valid, reason)
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            grayscale = image

        # Check intensity distribution with more lenient thresholds
        mean_intensity = np.mean(grayscale)
        std_intensity = np.std(grayscale)

        logger.info(f"Image characteristics - Mean: {mean_intensity}, Std: {std_intensity}")

        # Relaxed intensity checks
        if mean_intensity < 10:
            return False, "Image is too dark for a CT scan"
        if mean_intensity > 245:
            return False, "Image is too bright for a CT scan"
        if std_intensity < 5:
            return False, "Image lacks the contrast typical of CT scans"

        # Check for circular/oval structure (typical of chest CTs)
        _, binary = cv2.threshold(grayscale, mean_intensity, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, "No distinct structures found in the image"

        # More lenient aspect ratio check
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h

        logger.info(f"Contour aspect ratio: {aspect_ratio}")

        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False, "Image shape is not typical of chest CT scans"

        # More lenient edge density check
        edges = cv2.Canny(grayscale, 50, 150)  # Reduced thresholds
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        logger.info(f"Edge density: {edge_density}")

        if edge_density < 0.02:  # Reduced threshold
            return False, "Image lacks the detailed structure typical of CT scans"

        return True, "Valid CT scan image"

    except Exception as e:
        logger.error(f"Error in check_ct_characteristics: {str(e)}")
        return False, f"Error analyzing image characteristics: {str(e)}"

def enhance_contrast(image):
    """
    Enhance contrast of CT scan image.

    Args:
        image: numpy array of image

    Returns:
        numpy array: Contrast enhanced image
    """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)

        # Convert back to RGB for model input
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced
    except Exception as e:
        logger.error(f"Error in enhance_contrast: {str(e)}")
        raise

def validate_image(image):
    """
    Validate if the input is a valid CT scan image.

    Args:
        image: PIL Image or numpy array

    Returns:
        tuple: (bool, str) - (is_valid, message)
    """
    try:
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Basic dimension checks
        if not isinstance(image, np.ndarray):
            return False, "Invalid image format"

        # Check if grayscale or RGB
        if len(image.shape) not in [2, 3]:
            return False, "Invalid image dimensions"

        # If RGB, check channels
        if len(image.shape) == 3 and image.shape[2] not in [1, 3]:
            return False, "Invalid number of color channels"

        # Check if image has reasonable size
        if image.shape[0] < 32 or image.shape[1] < 32:
            return False, "Image resolution too low"

        # Check CT scan specific characteristics
        is_valid_ct, reason = check_ct_characteristics(image)
        return is_valid_ct, reason

    except Exception as e:
        logger.error(f"Error in validate_image: {str(e)}")
        return False, "Error validating image"

def preprocess_image(image):
    """
    Preprocess the input CT scan image for model inference.

    Args:
        image: PIL Image or numpy array

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Apply CT scan specific preprocessing
        enhanced = enhance_contrast(image)
        enhanced_pil = Image.fromarray(enhanced)

        # Define preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize(TARGET_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])

        # Apply preprocessing
        image_tensor = preprocess(enhanced_pil)
        return image_tensor.unsqueeze(0)

    except Exception as e:
        logger.error(f"Error during image preprocessing: {str(e)}")
        raise ValueError(f"Error during image preprocessing: {str(e)}")