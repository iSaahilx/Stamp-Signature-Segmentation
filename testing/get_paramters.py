import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np
import os
import glob
from datetime import datetime

def extract_signatures(image_path, output_path=None, use_dynamic_params=True):
    """
    Extracts signatures from a document image using Ahmet Ozlu's method
    
    Args:
        image_path: Path to the input document image
        output_path: Path to save the output binary mask (optional)
        use_dynamic_params: Whether to use dynamic model parameters or constant ones
        
    Returns:
        Binary mask image with signatures
    """
    # Read the image and convert to grayscale
    img = cv2.imread(image_path, 0)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Apply threshold to ensure binary
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Connected component analysis
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    
    # Set parameters based on mode
    if use_dynamic_params:
        # Dynamic parameters (calculated based on image properties)
        constant_parameter_1 = 84
        constant_parameter_2 = 250
        constant_parameter_3 = 100
        constant_parameter_4 = 18
    else:
        # Constant parameters (more aggressive for signature detection)
        constant_parameter_1 = 100
        constant_parameter_2 = 200
        constant_parameter_3 = 120
        constant_parameter_4 = 20
    
    # Calculate region properties
    the_biggest_component = 0
    total_area = 0
    counter = 0
    
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        
        # Track the biggest component
        if region.area >= 250 and region.area > the_biggest_component:
            the_biggest_component = region.area
    
    # Calculate average area of components
    average = total_area / counter if counter > 0 else 0
    
    # Calculate thresholds for outlier removal
    a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4
    
    # Remove small connected components
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    
    # Remove large connected components (like tables, etc.)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outliar_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0
    
    # Save pre-version temporarily
    temp_file = os.path.join(os.path.dirname(output_path if output_path else '.'), 'temp_pre_version.png')
    plt.imsave(temp_file, pre_version)
    
    # Read and ensure binary
    result = cv2.imread(temp_file, 0)
    result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Remove temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Save output if path is provided
    if output_path:
        cv2.imwrite(output_path, result)
    
    return result

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
            dynamic_output_path = os.path.join(output_folder_dynamic, f"{filename}_signature_mask.png")
            dynamic_result = extract_signatures(img_path, dynamic_output_path, use_dynamic_params=True)
            
            # Count non-zero pixels as a measure of detected signature areas
            dynamic_signature_pixels = np.count_nonzero(dynamic_result)
            dynamic_results.append((filename, dynamic_signature_pixels))
            
            # Process with constant parameters
            constant_output_path = os.path.join(output_folder_constant, f"{filename}_signature_mask.png")
            constant_result = extract_signatures(img_path, constant_output_path, use_dynamic_params=False)
            
            # Count non-zero pixels as a measure of detected signature areas
            constant_signature_pixels = np.count_nonzero(constant_result)
            constant_results.append((filename, constant_signature_pixels))
            
            print(f"Processed {filename}: Found {dynamic_signature_pixels} pixels (dynamic) and {constant_signature_pixels} pixels (constant)")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Generate summary report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(os.path.join(output_folder_dynamic, f"summary_dynamic_{timestamp}.txt"), 'w') as f:
        f.write("Summary Report - Dynamic Parameters\n")
        f.write("="*50 + "\n")
        f.write(f"Processed {len(image_files)} images\n")
        f.write("-"*50 + "\n")
        for filename, pixels in dynamic_results:
            f.write(f"{filename}: {pixels} signature pixels detected\n")
    
    with open(os.path.join(output_folder_constant, f"summary_constant_{timestamp}.txt"), 'w') as f:
        f.write("Summary Report - Constant Parameters\n")
        f.write("="*50 + "\n")
        f.write(f"Processed {len(image_files)} images\n")
        f.write("-"*50 + "\n")
        for filename, pixels in constant_results:
            f.write(f"{filename}: {pixels} signature pixels detected\n")
    
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
        dynamic_total = sum(pixels for _, pixels in dynamic_results)
        constant_total = sum(pixels for _, pixels in constant_results)
        
        print("\n=== Summary Statistics ===")
        print(f"Dynamic parameters: {dynamic_total} signature pixels detected")
        print(f"Constant parameters: {constant_total} signature pixels detected")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()