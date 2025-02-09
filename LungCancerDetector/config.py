"""Configuration settings for the lung cancer detection system."""

# Model parameters
MODEL_INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2

# Data preprocessing
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.dicom', '.dcm']
TARGET_SIZE = (224, 224)

# Model paths
MODEL_WEIGHTS_URL = "https://download.pytorch.org/models/resnet50-19c8e357.pth"

# Visualization
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES = ['Normal', 'Cancer']

# Image preprocessing
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
