import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
import os
import glob

# Parameters you can easily tweak - these are now global constants
# INCREASE these values to make detection MORE sensitive
GLOBAL_SENSITIVITY = 1   # Master sensitivity multiplier (higher = more sensitive)
MIN_THRESHOLD = 50         # Lower this to detect fainter signatures (range 0-255)
ADAPTIVE_BLOCK_SIZE = 3  # Lower this for more detail (must be odd: 3, 5, 7, 9, 11...)
ADAPTIVE_C_VALUE = 4       # Lower this to detect more faint details
CONNECT_SIZE = 4           # Increase to connect more signature parts
MIN_COMPONENT_AREA = 1     # Lower to keep smaller parts of signatures
SMALL_OBJECTS_FACTOR = 0.8 # Lower to keep more small components

def calculate_dynamic_parameters(image_size):
    # Adjust parameters based on image resolution
    total_pixels = image_size[0] * image_size[1]
    
    # HIGHLY sensitive scaling factors - exponential for very small images
    if total_pixels < 250000:  # Very small images (below ~0.25MP)
        scale_factor = np.sqrt(total_pixels) / 300 * GLOBAL_SENSITIVITY
    elif total_pixels < 500000:  # Small images (~0.5MP)
        scale_factor = np.sqrt(total_pixels) / 300 * GLOBAL_SENSITIVITY
    else:
        scale_factor = np.sqrt(total_pixels) / 500 * GLOBAL_SENSITIVITY
    
    return {
        'threshold': min(200, max(MIN_THRESHOLD, int(100 - (40 * (1 - scale_factor))))),
        'param1': max(10, int(40 * scale_factor)),      # Base divisor - LOWER = more sensitive
        'param2': max(25, int(60 * scale_factor)),      # Size threshold multiplier
        'param3': max(5, int(20 * scale_factor)),       # Minimum size threshold - LOWER = more sensitive
        'param4': max(20, int(15 * scale_factor)),      # Maximum size multiplier - HIGHER = more sensitive
        'min_area': max(MIN_COMPONENT_AREA, int(4 * scale_factor)),
        'small_tolerance': max(SMALL_OBJECTS_FACTOR, 0.3 - (0.2 * (1 - scale_factor))),
        'large_tolerance': min(3.0, 2.0 + (0.5 * (1 - scale_factor))),
        'connect_size': max(1, min(5, int(CONNECT_SIZE * scale_factor)))
    }

def preprocess_image(img, params):
    # Make a copy to avoid modifying the original
    processed = img.copy()
    
    # Apply contrast enhancement - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    
    # Apply bilateral filter to preserve edges while reducing noise
    processed = cv2.bilateralFilter(processed, 5, 75, 75)
    
    # Apply adaptive thresholding - MUCH more sensitive to details
    adaptive = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C_VALUE
    )
    
    # Also try global thresholding (Otsu + additional threshold)
    _, global_thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Additional threshold for very faint signatures
    _, faint_thresh = cv2.threshold(processed, params['threshold'], 255, cv2.THRESH_BINARY_INV)
    
    # Combine all approaches - this makes it VERY sensitive
    combined = cv2.bitwise_or(cv2.bitwise_or(adaptive, global_thresh), faint_thresh)
    
    # Connect close components (crucial for signatures)
    kernel_size = params['connect_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    # Dilate slightly to make components more prominent
    combined = cv2.dilate(combined, np.ones((2, 2), np.uint8), iterations=1)
    
    return combined

# Input and output directories
input_dir = r"D:\Hackathon\images\pro"
output_dir = r"D:\Hackathon\images\pro\output"
pre_version_dir = r"D:\Hackathon\images\output"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pre_version_dir, exist_ok=True)

# Get all jpg images from input directory
input_images = glob.glob(os.path.join(input_dir, "*.jpg"))

for input_image_path in input_images:
    try:
        base_name = os.path.basename(input_image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        print(f"Processing {base_name}...")
        
        # Read the input image
        img = cv2.imread(input_image_path, 0)
        if img is None:
            print(f"Could not read image: {input_image_path}")
            continue
        
        # Store original dimensions for reporting
        original_height, original_width = img.shape
        total_pixels = original_height * original_width
        
        # Get dynamic parameters based on image size
        params = calculate_dynamic_parameters(img.shape)
        
        # Enhanced preprocessing - now with calculated parameters
        img_processed = preprocess_image(img, params)
        
        # Save intermediate result for debugging
        debug_path = os.path.join(pre_version_dir, f"debug_{name_without_ext}.png")
        cv2.imwrite(debug_path, img_processed)
        
        # Connected component analysis
        blobs = img_processed > 0
        blobs_labels = measure.label(blobs, background=0)
        
        # Calculate statistics for components
        areas = []
        max_component = 0
        
        for region in regionprops(blobs_labels):
            if region.area > params['min_area']:
                areas.append(region.area)
            if region.area > max_component:
                max_component = region.area
        
        # Use median rather than mean for better robustness
        if areas:
            median_area = np.median(areas)
            average = median_area  # Use median for more robustness
        else:
            average = max_component / 2 if max_component > 0 else 50
        
        # Calculate thresholds with adaptive tolerance - MORE SENSITIVE
        small_size_threshold = ((average / params['param1']) * params['param2']) + params['param3']
        large_size_threshold = small_size_threshold * params['param4']
        
        # Print diagnostics
        print(f"  Image size: {original_width}x{original_height} ({total_pixels} pixels)")
        print(f"  Component stats: {len(areas)} components, median size: {average:.2f}")
        print(f"  Thresholds: small={small_size_threshold:.2f}, large={large_size_threshold:.2f}")
        print(f"  Sensitivity parameters: global={GLOBAL_SENSITIVITY}, min_thresh={MIN_THRESHOLD}")
        
        # Remove components with highly adaptive tolerance
        pre_version = morphology.remove_small_objects(
            blobs_labels, 
            min_size=max(3, int(small_size_threshold * params['small_tolerance']))
        )
        
        # Handle large components with adaptive tolerance
        component_sizes = np.bincount(pre_version.ravel())
        too_large = component_sizes > large_size_threshold * params['large_tolerance']
        too_large_mask = too_large[pre_version]
        pre_version[too_large_mask] = 0
        
        # Save pre-version (normalized to 0-255 for visualization)
        pre_version_normalized = (pre_version > 0) * 255
        pre_version_path = os.path.join(pre_version_dir, f"pre_version_{name_without_ext}.png")
        plt.imsave(pre_version_path, pre_version_normalized, cmap='gray')
        
        # Final processing - we'll use the pre-version directly rather than re-reading
        # and just invert it for final output format
        final_img = 255 - pre_version_normalized.astype(np.uint8)
        
        # Apply a final cleanup with a small closing operation to connect nearby components
        kernel = np.ones((2, 2), np.uint8)
        final_img = cv2.morphologyEx(final_img, cv2.MORPH_CLOSE, kernel)
        
        # Save result
        output_path = os.path.join(output_dir, f"{name_without_ext}_out.png")
        cv2.imwrite(output_path, final_img)
        print(f"  Saved output to {output_path}")
        print("  Done!")
    
    except Exception as e:
        print(f"Error processing {input_image_path}: {str(e)}")

print("Processing complete!")