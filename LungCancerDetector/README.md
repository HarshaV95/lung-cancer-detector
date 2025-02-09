git clone <repository-url>
   cd lung-cancer-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python -m streamlit run app.py
   ```

4. Access the interface:
   - Open your web browser
   - Navigate to `http://localhost:8501`

## Project Structure

```
├── app.py                # Main Streamlit application
├── config.py            # Configuration settings
├── model.py             # ResNet model implementation
├── preprocessing.py     # Image preprocessing pipeline
├── utils.py            # Utility functions
├── test_image_generator.py # Test image generation
└── docs/
    ├── TECHNICAL.md    # Technical documentation
    └── USER_GUIDE.md   # User guide
```

## Usage

1. Start the web interface:
   ```bash
   python -m streamlit run app.py
   ```

2. Upload a chest CT scan:
   - Click "Choose a chest CT scan image..."
   - Select a valid CT scan image (JPG, PNG formats supported)
   - Wait for validation and analysis

3. Review Results:
   - Prediction (Normal/Cancer)
   - Confidence score
   - Risk level indicator (color-coded)
   - Detailed analysis explanation

## Testing

The project includes a test image generator for validation:

```bash
python test_image_generator.py