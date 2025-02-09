import streamlit as st
import torch
from PIL import Image
import io
import logging
import sys
from pathlib import Path
import traceback

from model import load_model, predict
from preprocessing import preprocess_image, validate_image
from utils import is_valid_file, format_confidence, get_result_color
from config import CLASS_NAMES, CONFIDENCE_THRESHOLD

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Verify environment setup and dependencies."""
    try:
        logger.info("Verifying dependencies...")
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        import torchvision
        logger.info(f"TorchVision version: {torchvision.__version__}")
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        return False

# Ensure the upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@st.cache_resource(ttl=3600)
def load_cached_model():
    return load_model()

def main():
    try:
        logger.info("Starting application setup...")
        if not setup_environment():
            st.error("⚠️ Failed to initialize required dependencies. Please check the logs.")
            return

        # Configure Streamlit
        st.set_page_config(
            page_title="Lung Cancer Detection System",
            page_icon="🫁",
            layout="wide",
            initial_sidebar_state="collapsed"
        )

        logger.info("Starting Lung Cancer Detection System")

        st.title("Lung Cancer Detection System")
        st.write("""
        Upload a 2D CT scan image to check for potential lung cancer indicators.
        The system uses a ResNet-based deep learning model for analysis.
        """)

        st.info("""
        ℹ️ **Important**: This system only works with chest CT scan images. 
        Regular photos or other medical images will not be accepted.
        """)

        try:
            logger.info("Loading model...")
            model = load_cached_model()
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            st.error("⚠️ Error loading the model. Please try refreshing the page.")
            st.stop()
            return

        uploaded_file = st.file_uploader(
            "Choose a chest CT scan image...",
            type=['png', 'jpg', 'jpeg'],
            key='ct_scan_uploader'
        )

        if uploaded_file is not None:
            try:
                logger.info(f"Processing uploaded file: {uploaded_file.name}")

                if not is_valid_file(uploaded_file.name):
                    st.error("Please upload a valid image file (PNG, JPG, or JPEG).")
                    return

                image_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(image_bytes))

                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                is_valid, validation_message = validate_image(image)
                if not is_valid:
                    st.error(f"❌ Invalid image: {validation_message}")
                    st.warning("""
                    Please ensure you're uploading a chest CT scan image. 
                    Regular photos or other types of medical images are not supported.
                    """)
                    return

                with st.spinner("Analyzing CT scan..."):
                    logger.info("Preprocessing image...")
                    image_tensor = preprocess_image(image)
                    logger.info("Making prediction...")
                    predicted_class, confidence = predict(model, image_tensor)
                    logger.info(f"Prediction complete: class={predicted_class}, confidence={confidence}")

                with col2:
                    st.subheader("Analysis Results")

                    result_color = get_result_color(confidence)

                    if confidence < 0.7:
                        st.warning("⚠️ Low confidence prediction. Please ensure the image is a clear chest CT scan.")

                    st.markdown(f"""
                    <div style='padding: 20px; border-radius: 5px; background-color: {result_color}20;'>
                        <h3 style='color: {result_color}'>
                            {CLASS_NAMES[predicted_class]}
                        </h3>
                        <p>Confidence: {format_confidence(confidence)}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if predicted_class == 1 and confidence > CONFIDENCE_THRESHOLD:
                        st.warning("""
                        ⚠️ Potential cancer indicators detected. Please consult with a healthcare professional
                        for proper diagnosis. This is an automated screening tool and should not be used as
                        a definitive diagnosis.
                        """)

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
                st.error("⚠️ An error occurred while processing the image.")
                st.error("Please ensure you're uploading a valid chest CT scan image.")

        st.markdown("""
        ### About the System
        - This system is designed specifically for chest CT scan analysis
        - Uses advanced image processing to validate and analyze CT scans
        - Provides confidence scores for predictions
        - For screening purposes only - not a substitute for professional medical diagnosis

        ### Valid Input Requirements
        - Must be a chest CT scan image
        - Common image formats (PNG, JPG, JPEG)
        - Clear, high-quality scans
        - Proper contrast and brightness levels
        """)

    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("⚠️ An unexpected error occurred. Please check the logs and refresh the page.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print("Error: Please use 'streamlit run app.py' to start the application")