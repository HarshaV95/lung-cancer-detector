import torch
import torch.nn as nn
from torchvision.models import resnet50
from config import NUM_CLASSES, MODEL_WEIGHTS_URL
import logging
import os

logger = logging.getLogger(__name__)

class LungCancerModel(nn.Module):
    def __init__(self, pretrained=True):
        """
        Initialize the lung cancer detection model based on ResNet50.

        Args:
            pretrained (bool): Whether to use pretrained weights
        """
        super(LungCancerModel, self).__init__()

        try:
            logger.info("Initializing ResNet50 model...")
            # Load base ResNet model
            resnet = resnet50(weights='IMAGENET1K_V1' if pretrained else None)

            # Modify first conv layer to better handle medical images
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                # Average the weights across channels for grayscale
                self.conv1.weight.data = torch.mean(resnet.conv1.weight.data, dim=1, keepdim=True).repeat(1,3,1,1)

            # Keep most of ResNet backbone
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.avgpool = resnet.avgpool

            # Add medical imaging specific head
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, NUM_CLASSES)
            )

            logger.info("Model architecture modified for medical imaging")

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def forward(self, x):
        """Forward pass of the model."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def load_model():
    """
    Load the model with pretrained weights.

    Returns:
        LungCancerModel: Initialized model
    """
    try:
        logger.info("Setting up device...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        logger.info("Initializing model...")
        model = LungCancerModel(pretrained=True)
        model = model.to(device)
        model.eval()
        logger.info("Model initialized and set to evaluation mode")

        return model

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}", exc_info=True)
        raise

def predict(model, image_tensor):
    """
    Make prediction on preprocessed image.

    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor

    Returns:
        tuple: (prediction class, confidence score)
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

            # Add additional threshold for medical predictions
            if confidence < 0.7:  # Higher threshold for medical diagnosis
                logger.warning("Low confidence prediction")

        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Error in predict: {str(e)}", exc_info=True)
        raise