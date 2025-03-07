import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
import os
import glob
from datetime import datetime

def detect_and_mask_signature(image_path, output_path=None, use_dynamic_params=True):
    """
    Detects signatures in a document image and masks them out (replaces with white)
    
    Args:
        image_path: Path to the input document image
        output_path: Path to save the output image (optional)
        use_dynamic_params: Whether to use dynamic model parameters or constant ones
        
    Returns:
        Tuple of (original image, masked image, signature regions)
    """
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    # Make a copy for masking
    masked_img = original.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilate to connect signature components
    dilated = cv2.dilate(opening, kernel, iterations=2)
    
    # Label connected components
    labels = measure.label(dilated)
    
    # Measure properties of connected components
    regions = measure.regionprops(labels)
    
    # Store potential signature regions
    signature_regions = []
    
    # Set parameters based on mode
    if use_dynamic_params:
        # Dynamic parameters from the model
        min_region_size = 20
        min_complexity = 70
        density_range = (0.05, 0.4)
        min_stroke_variation = 0.5
        aspect_ratio_range = (0.2, 5)
        alt_width_threshold = 100
        alt_complexity = 60
        alt_density_range = (0.05, 0.3)
    else:
        # Constant parameters
        min_region_size = 15  # More sensitive to smaller signatures
        min_complexity = 50   # Less strict on complexity
        density_range = (0.03, 0.5)  # Wider density range
        min_stroke_variation = 0.4   # Less strict on stroke variation
        aspect_ratio_range = (0.1, 6)  # Wider aspect ratio range
        alt_width_threshold = 80
        alt_complexity = 45
        alt_density_range = (0.03, 0.4)
    
    for region in regions:
        # Get region coordinates
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        
        # Filter small noise
        if width < min_region_size or height < min_region_size:
            continue
            
        # Extract region of interest
        roi = dilated[minr:maxr, minc:maxc]
        
        # Calculate signature-specific metrics
        
        # 1. Curvature analysis
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip regions with no contours
        if not contours:
            continue
            
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate complexity (perimeter²/area)
        perimeter = cv2.arcLength(largest_contour, True)
        area = cv2.contourArea(largest_contour)
        
        complexity = 0
        if area > 0:
            complexity = (perimeter * perimeter) / area
        
        # 2. Calculate density
        density = np.sum(roi) / (roi.shape[0] * roi.shape[1] * 255)
        
        # 3. Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        
        # 4. Analyze stroke width variation
        horizontal_projection = np.sum(roi, axis=1)
        vertical_projection = np.sum(roi, axis=0)
        h_variation = np.std(horizontal_projection) / (np.mean(horizontal_projection) + 1e-10)
        v_variation = np.std(vertical_projection) / (np.mean(vertical_projection) + 1e-10)
        stroke_variation = (h_variation + v_variation) / 2
        
        # Combine metrics for signature classification
        is_signature = (complexity > min_complexity and 
                        density_range[0] < density < density_range[1] and 
                        stroke_variation > min_stroke_variation and
                        aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1])
        
        # Alternative condition for longer signatures
        if width > alt_width_threshold and complexity > alt_complexity and alt_density_range[0] < density < alt_density_range[1]:
            is_signature = True
            
        if is_signature:
            # Store coordinates for masking
            signature_regions.append((minc, minr, maxc, maxr))
            
            # Draw rectangle on original image (for visualization)
            cv2.rectangle(original, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            
            # Mask out the signature region (replace with white)
            masked_img[minr:maxr, minc:maxc] = [255, 255, 255]
    
    # Save output if path is provided
    if output_path:
        cv2.imwrite(output_path, masked_img)
    
    return original, masked_img, signature_regions

def process_folder(input_folder, output_folder_dynamic, output_folder_constant):
    """
    Processes all images in a folder using both parameter sets
    
    Args:
        input_folder: Path to the folder containing images
        output_folder_dynamic: Path to save images processed with dynamic parameters
        output_folder_constant: Path to save images processed with constant parameters
    """
    # Create output folders if they don't exist
    os.makedirs(output_folder_dynamic, exist_ok=True)
    os.makedirs(output_folder_constant, exist_ok=True)
    
    # Get all image files in the input folder
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    # Process each image
    dynamic_results = []
    constant_results = []
    
    for img_path in image_files:
        try:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(img_path))[0]
            
            # Process with dynamic parameters
            dynamic_output_path = os.path.join(output_folder_dynamic, f"{filename}_masked.png")
            _, _, dynamic_regions = detect_and_mask_signature(img_path, dynamic_output_path, use_dynamic_params=True)
            dynamic_results.append((filename, len(dynamic_regions)))
            
            # Process with constant parameters
            constant_output_path = os.path.join(output_folder_constant, f"{filename}_masked.png")
            _, _, constant_regions = detect_and_mask_signature(img_path, constant_output_path, use_dynamic_params=False)
            constant_results.append((filename, len(constant_regions)))
            
            print(f"Processed {filename}: Found {len(dynamic_regions)} regions (dynamic) and {len(constant_regions)} regions (constant)")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Generate summary report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(os.path.join(output_folder_dynamic, f"summary_dynamic_{timestamp}.txt"), 'w') as f:
        f.write("Summary Report - Dynamic Parameters\n")
        f.write("="*50 + "\n")
        f.write(f"Processed {len(image_files)} images\n")
        f.write("-"*50 + "\n")
        for filename, count in dynamic_results:
            f.write(f"{filename}: {count} signature regions\n")
    
    with open(os.path.join(output_folder_constant, f"summary_constant_{timestamp}.txt"), 'w') as f:
        f.write("Summary Report - Constant Parameters\n")
        f.write("="*50 + "\n")
        f.write(f"Processed {len(image_files)} images\n")
        f.write("-"*50 + "\n")
        for filename, count in constant_results:
            f.write(f"{filename}: {count} signature regions\n")
    
    return len(image_files), dynamic_results, constant_results

def main():

    input_folder = "D:/Hackathon/images/preprocessed without augmentation"
    output_folder_dynamic = "D:/Hackathon/output/dynamic"
    output_folder_constant = "D:/Hackathon/output/constant"
    
    try:
        num_processed, dynamic_results, constant_results = process_folder(
            input_folder, output_folder_dynamic, output_folder_constant)
        
        print(f"\nProcessing complete! Processed {num_processed} images.")
        print(f"Results saved to: \n- {output_folder_dynamic} \n- {output_folder_constant}")
        
        # Display summary statistics
        dynamic_total = sum(count for _, count in dynamic_results)
        constant_total = sum(count for _, count in constant_results)
        
        print("\n=== Summary Statistics ===")
        print(f"Dynamic parameters: {dynamic_total} signature regions detected")
        print(f"Constant parameters: {constant_total} signature regions detected")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()