# Signature Extraction Tool - Detailed Usage Guide

## Getting Started

### Running the Application

1. Open a terminal or command prompt
2. Navigate to the directory containing the script
3. Run the application:
   ```bash
   python signature_extractor.py
   ```

### Main Window Overview

When the application launches, you'll see the main interface with these components:
- Top section with control buttons
- Original document display area
- Tabbed result panels
- Status bar at the bottom

## Step-by-Step Usage

### 1. Loading a Document

1. Click the **Upload Image** button in the top-right corner
2. In the file dialog that appears, navigate to and select your document image
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF
3. Once selected, a preview of the document will appear in the "Original Document" section
4. The status bar will confirm successful loading with "Image loaded: [filename]"

### 2. Extracting Signatures

1. After loading an image, the **Extract Signatures** button becomes available
2. Click this button to begin processing
3. The status bar will show progress messages during processing
4. Processing may take a few seconds depending on image size and complexity

### 3. Reviewing Results

After processing, results appear in four tabs:

#### Dynamic Parameters Tab

Shows extraction results using adaptive parameters:
- Left panel: Binary mask showing detected signature areas
- Right panel: Extracted signatures on white background
- Bottom text: Details of parameters used and pixel counts

#### Static Parameters Tab

Shows extraction results using fixed parameters:
- Left panel: Binary mask showing detected signature areas
- Right panel: Extracted signatures on white background
- Bottom text: Details of parameters used and pixel counts

#### Inverted Mask Tab

Shows document with signatures removed:
- Left panel: Document with signatures removed using dynamic parameters
- Right panel: Document with signatures removed using static parameters
- This view is useful for document content extraction or OCR

#### Comparison Tab

Provides detailed analysis comparing the two extraction methods:
- Parameter values used in each method
- Pixel count differences
- Percentage difference
- Recommendation on which method performed better

### 4. Saving Results

1. Click the **Save Signatures** button
2. Select a directory where you want to save the extracted images
3. The application will save four files:
   - `dynamic_signature.png`: Signatures extracted using dynamic parameters
   - `static_signature.png`: Signatures extracted using static parameters
   - `dynamic_inverted.png`: Document with signatures removed (dynamic parameters)
   - `static_inverted.png`: Document with signatures removed (static parameters)
4. A confirmation dialog will appear when saving is complete

## Advanced Usage

### Interpreting the Parameter Values

In the Comparison tab, you'll see these key metrics:

- **signature_pixels**: The number of pixels identified as part of signatures
- **constant_parameter_1**: Controls the small object size threshold (lower = more sensitive)
- **constant_parameter_2**: Affects size outlier threshold calculation
- **constant_parameter_3**: Base threshold value for small objects
- **constant_parameter_4**: Multiplier for large object threshold

The method that extracts more signature pixels generally provides better results, but visual inspection is still recommended.

### Working with Different Document Types

- **Text-heavy documents**: Dynamic parameters often work better
- **Forms and structured documents**: Static parameters may be more effective
- **Low contrast documents**: Results may vary; compare both methods

### Batch Processing

While the GUI doesn't support batch processing directly, you can:
1. Extract and save the `extract_signatures()`, `extract_signature_area()`, and `extract_inverted_area()` functions
2. Create a simple script that loops through files in a directory
3. Process each file and save results to an output directory

Example batch processing script:

```python
import os
import cv2
import numpy as np
from signature_extraction_functions import extract_signatures, extract_signature_area

input_dir = "input_documents"
output_dir = "extracted_signatures"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        # Load image
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        mask, params, pixels = extract_signatures(image, use_dynamic_params=True)
        signature = extract_signature_area(image, mask)
        
        # Save result
        output_path = os.path.join(output_dir, f"sig_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(signature, cv2.COLOR_RGB2BGR))
        
        print(f"Processed {filename} - Found {pixels} signature pixels")
```

## Troubleshooting

### Common Issues and Solutions

1. **"Error loading image"**
   - Ensure the file is a valid image format
   - Check if the file is corrupted or incomplete
   - Try converting the image to another format (e.g., PNG)

2. **No signatures detected (blank result)**
   - The document may have very light signatures
   - Try scanning the document at a higher resolution
   - Adjust contrast in the scanned document 

3. **Too much noise in extracted signatures**
   - The algorithm may be picking up text as signatures
   - Try using the static parameters instead of dynamic
   - Consider pre-processing the document to increase contrast between signatures and text

4. **Application freezes during processing**
   - Large images require more processing time and memory
   - Try resizing very large images before processing
   - Ensure your system has sufficient memory

5. **Missing libraries errors**
   - Install all required dependencies:
     ```bash
     pip install opencv-python numpy scikit-image matplotlib pillow
     ```

### Performance Optimization

- For faster processing, consider downscaling large images before uploading
- Processing speed depends primarily on image resolution, not content complexity
- The application uses temporary files during processing; ensure adequate disk space