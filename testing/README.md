# Signature Extraction Tool

A simple web-based tool for extracting signatures from document images using OpenCV and Streamlit.

## Features
- Supports multiple image formats (`JPG, JPEG, PNG, BMP, TIFF`)
- Uses connected component analysis for signature extraction
- Allows dynamic and static parameter-based extraction
- Displays extracted signatures for comparison
- Easy deployment using Streamlit

## Installation
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage
Run the application with:

```bash
streamlit run app.py
```

Upload a document image, click **Extract Signatures**, and view the results.

## Dependencies
- Python 3.x
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- scikit-image
- Pillow

## Deployment
To deploy on Streamlit Cloud:
1. Push the code to **GitHub**
2. Connect your repository to **Streamlit Cloud**
3. Click **Deploy**

