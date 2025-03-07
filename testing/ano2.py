import cv2
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
import os
import tempfile
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Signature extraction function
def extract_signatures(image, use_dynamic_params=True):
    """
    Extracts signatures from a document image using Ahmet Ozlu's method.
    """
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    
    if use_dynamic_params:
        constant_parameter_1 = 84
        constant_parameter_2 = 250
        constant_parameter_3 = 100
        constant_parameter_4 = 18
    else:
        constant_parameter_1 = 100
        constant_parameter_2 = 200
        constant_parameter_3 = 120
        constant_parameter_4 = 20
    
    the_biggest_component = 0
    total_area = 0
    counter = 0
    
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250 and region.area > the_biggest_component:
            the_biggest_component = region.area
    
    average = total_area / counter if counter > 0 else 0
    a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4
    
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outliar_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        plt.imsave(temp_path, pre_version)
    
    result = cv2.imread(temp_path, 0)
    result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    signature_pixels = np.count_nonzero(result)
    
    params = {
        "constant_parameter_1": constant_parameter_1,
        "constant_parameter_2": constant_parameter_2,
        "constant_parameter_3": constant_parameter_3,
        "constant_parameter_4": constant_parameter_4,
        "signature_pixels": signature_pixels
    }
    
    return result, params, signature_pixels


def extract_signature_area(original_image, mask):
    """Extracts the signature from the original image using the mask."""
    if len(original_image.shape) == 3:
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        mask_inv = cv2.bitwise_not(mask_3c)
        signature_only = cv2.bitwise_and(original_image, mask_inv)
        white_bg = np.ones_like(original_image) * 255
        bg_mask = cv2.bitwise_not(mask_inv)
        final_result = cv2.bitwise_and(white_bg, bg_mask) + signature_only
        return final_result
    else:
        mask_inv = cv2.bitwise_not(mask)
        signature_only = cv2.bitwise_and(original_image, mask_inv)
        white_bg = np.ones_like(original_image) * 255
        bg_mask = cv2.bitwise_not(mask_inv)
        final_result = cv2.bitwise_and(white_bg, bg_mask) + signature_only
        return final_result


# Tkinter GUI Application
class SignatureExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Extraction Tool")
        self.root.geometry("800x600")
        
        # Variables
        self.image = None
        self.dynamic_mask = None
        self.static_mask = None
        self.dynamic_signature = None
        self.static_signature = None
        
        # UI Elements
        self.label = tk.Label(root, text="Upload a document image to extract signatures", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)
        
        self.process_button = tk.Button(root, text="Extract Signatures", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=10)
        
        self.tab_control = ttk.Notebook(root)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="Dynamic Parameters")
        self.tab_control.add(self.tab2, text="Static Parameters")
        self.tab_control.add(self.tab3, text="Comparison")
        self.tab_control.pack(expand=1, fill="both")
        
        # Dynamic Parameters Tab
        self.dynamic_mask_label = tk.Label(self.tab1, text="Signature Mask")
        self.dynamic_mask_label.pack()
        self.dynamic_signature_label = tk.Label(self.tab1, text="Extracted Signature")
        self.dynamic_signature_label.pack()
        
        # Static Parameters Tab
        self.static_mask_label = tk.Label(self.tab2, text="Signature Mask")
        self.static_mask_label.pack()
        self.static_signature_label = tk.Label(self.tab2, text="Extracted Signature")
        self.static_signature_label.pack()
        
        # Comparison Tab
        self.comparison_label = tk.Label(self.tab3, text="Dynamic vs Static Parameters")
        self.comparison_label.pack()
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.process_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Image uploaded successfully!")
    
    def process_image(self):
        if self.image is not None:
            # Process with dynamic parameters
            self.dynamic_mask, dynamic_params, dynamic_pixels = extract_signatures(self.image, use_dynamic_params=True)
            self.dynamic_signature = extract_signature_area(self.image, self.dynamic_mask)
            
            # Process with static parameters
            self.static_mask, static_params, static_pixels = extract_signatures(self.image, use_dynamic_params=False)
            self.static_signature = extract_signature_area(self.image, self.static_mask)
            
            # Display results
            self.display_results()
    
    def display_results(self):
        # Display dynamic results
        dynamic_mask_img = Image.fromarray(self.dynamic_mask)
        dynamic_mask_img = dynamic_mask_img.resize((300, 300))
        dynamic_mask_img = ImageTk.PhotoImage(dynamic_mask_img)
        self.dynamic_mask_label.config(image=dynamic_mask_img)
        self.dynamic_mask_label.image = dynamic_mask_img
        
        dynamic_signature_img = Image.fromarray(self.dynamic_signature)
        dynamic_signature_img = dynamic_signature_img.resize((300, 300))
        dynamic_signature_img = ImageTk.PhotoImage(dynamic_signature_img)
        self.dynamic_signature_label.config(image=dynamic_signature_img)
        self.dynamic_signature_label.image = dynamic_signature_img
        
        # Display static results
        static_mask_img = Image.fromarray(self.static_mask)
        static_mask_img = static_mask_img.resize((300, 300))
        static_mask_img = ImageTk.PhotoImage(static_mask_img)
        self.static_mask_label.config(image=static_mask_img)
        self.static_mask_label.image = static_mask_img
        
        static_signature_img = Image.fromarray(self.static_signature)
        static_signature_img = static_signature_img.resize((300, 300))
        static_signature_img = ImageTk.PhotoImage(static_signature_img)
        self.static_signature_label.config(image=static_signature_img)
        self.static_signature_label.image = static_signature_img


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureExtractorApp(root)
    root.mainloop()