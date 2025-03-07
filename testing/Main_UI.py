import cv2
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Signature extraction function
def extract_signatures(image, use_dynamic_params=True):
    """
    Extracts signatures from a document image using Ahmet Ozlu's method.
    """
    # Ensure image is grayscale
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Apply binary threshold
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Create blob mask where foreground is True and background is False
    blobs = img < img.mean()  # Fixed: invert the threshold comparison
    blobs_labels = measure.label(blobs, background=0)  # Fixed: background value
    
    # Set parameters based on mode
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
    
    # Analyze components
    the_biggest_component = 0
    total_area = 0
    counter = 0
    
    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250 and region.area > the_biggest_component:
            the_biggest_component = region.area
    
    # Calculate size thresholds
    average = total_area / counter if counter > 0 else 0
    a4_small_size_outlier_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outlier_constant = a4_small_size_outlier_constant * constant_parameter_4
    
    # Remove small objects
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outlier_constant)
    
    # Remove large objects (typically the document background)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outlier_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0
    
    # Save and load to convert to binary image format
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # Fixed: properly save the pre_version image
    plt.imsave(temp_path, pre_version, cmap='binary')
    
    result = cv2.imread(temp_path, 0)
    if result is None:
        # Ensure we have a valid result even if file operation failed
        result = np.zeros_like(img)
    else:
        result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Clean up temp file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"Warning: Could not remove temporary file: {e}")
    
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
    # Create a copy to avoid modifying the original
    original_copy = original_image.copy()
    
    if len(original_copy.shape) == 3:
        # For color images
        # Convert single-channel mask to 3-channel if needed
        if len(mask.shape) == 2:
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_3c = mask.copy()
            
        # Extract signature (black pixels on white background)
        white_bg = np.ones_like(original_copy) * 255
        
        # Create inverted mask - where signatures are
        mask_inv = cv2.bitwise_not(mask_3c)
        
        # Extract signature parts
        signature_only = cv2.bitwise_and(original_copy, original_copy, mask=cv2.bitwise_not(mask))
        
        # Create white background where there's no signature
        white_areas = cv2.bitwise_and(white_bg, white_bg, mask=mask)
        
        # Combine signature and white background
        final_result = cv2.add(signature_only, white_areas)
    else:
        # For grayscale images
        if len(mask.shape) == 3:
            # If mask is somehow 3-channel but original is grayscale, convert mask to single channel
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            
        # Create inverted mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract signature
        signature_only = cv2.bitwise_and(original_copy, mask_inv)
        
        # Create white background
        white_bg = np.ones_like(original_copy) * 255
        white_areas = cv2.bitwise_and(white_bg, mask)
        
        # Combine
        final_result = cv2.add(signature_only, white_areas)
    
    return final_result


# Tkinter GUI Application
class SignatureExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Extraction Tool")
        self.root.geometry("900x700")
        
        # Variables
        self.image = None
        self.original_display = None
        self.dynamic_mask = None
        self.static_mask = None
        self.dynamic_signature = None
        self.static_signature = None
        self.dynamic_params = None
        self.static_params = None
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # UI Elements - Top Section
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.label = ttk.Label(top_frame, text="Upload a document image to extract signatures", font=("Arial", 14))
        self.label.pack(side=tk.LEFT, padx=(0, 20))
        
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        self.upload_button = ttk.Button(btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.process_button = ttk.Button(btn_frame, text="Extract Signatures", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT)
        
        self.save_button = ttk.Button(btn_frame, text="Save Signatures", command=self.save_signatures, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Original image display
        self.original_frame = ttk.LabelFrame(main_frame, text="Original Document")
        self.original_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(padx=10, pady=10)
        
        # Tab control for results
        self.tab_control = ttk.Notebook(main_frame)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text="Dynamic Parameters")
        self.tab_control.add(self.tab2, text="Static Parameters")
        self.tab_control.add(self.tab3, text="Comparison")
        self.tab_control.pack(expand=True, fill="both")
        
        # Dynamic Parameters Tab
        dyn_frame = ttk.Frame(self.tab1, padding=10)
        dyn_frame.pack(fill=tk.BOTH, expand=True)
        
        dyn_left_frame = ttk.Frame(dyn_frame)
        dyn_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        dyn_right_frame = ttk.Frame(dyn_frame)
        dyn_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(dyn_left_frame, text="Signature Mask").pack(pady=(0, 5))
        self.dynamic_mask_label = ttk.Label(dyn_left_frame, borderwidth=1, relief="solid")
        self.dynamic_mask_label.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(dyn_right_frame, text="Extracted Signature").pack(pady=(0, 5))
        self.dynamic_signature_label = ttk.Label(dyn_right_frame, borderwidth=1, relief="solid")
        self.dynamic_signature_label.pack(fill=tk.BOTH, expand=True)
        
        self.dynamic_info_label = ttk.Label(self.tab1, text="", justify=tk.LEFT)
        self.dynamic_info_label.pack(pady=10, fill=tk.X)
        
        # Static Parameters Tab
        static_frame = ttk.Frame(self.tab2, padding=10)
        static_frame.pack(fill=tk.BOTH, expand=True)
        
        static_left_frame = ttk.Frame(static_frame)
        static_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        static_right_frame = ttk.Frame(static_frame)
        static_right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(static_left_frame, text="Signature Mask").pack(pady=(0, 5))
        self.static_mask_label = ttk.Label(static_left_frame, borderwidth=1, relief="solid")
        self.static_mask_label.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(static_right_frame, text="Extracted Signature").pack(pady=(0, 5))
        self.static_signature_label = ttk.Label(static_right_frame, borderwidth=1, relief="solid")
        self.static_signature_label.pack(fill=tk.BOTH, expand=True)
        
        self.static_info_label = ttk.Label(self.tab2, text="", justify=tk.LEFT)
        self.static_info_label.pack(pady=10, fill=tk.X)
        
        # Comparison Tab
        comp_frame = ttk.Frame(self.tab3, padding=10)
        comp_frame.pack(fill=tk.BOTH, expand=True)
        
        self.comparison_text = tk.Text(comp_frame, height=15, width=80)
        self.comparison_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif")]
            )
            if file_path:
                self.status_var.set(f"Loading image: {os.path.basename(file_path)}")
                self.root.update()
                
                # Load and convert image
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise ValueError("Failed to load image")
                    
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                
                # Display original image
                self.display_original()
                
                self.process_button.config(state=tk.NORMAL)
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Error loading image")
    
    def display_original(self):
        if self.image is not None:
            # Resize for display while maintaining aspect ratio
            h, w = self.image.shape[:2]
            max_height = 200
            if h > max_height:
                scale = max_height / h
                display_size = (int(w * scale), max_height)
            else:
                display_size = (w, h)
                
            display_img = cv2.resize(self.image, display_size)
            self.original_display = ImageTk.PhotoImage(Image.fromarray(display_img))
            self.original_label.config(image=self.original_display)
    
    def process_image(self):
        if self.image is None:
            messagebox.showerror("Error", "No image loaded")
            return
            
        try:
            self.status_var.set("Processing image with dynamic parameters...")
            self.root.update()
            
            # Process with dynamic parameters
            self.dynamic_mask, self.dynamic_params, dynamic_pixels = extract_signatures(
                self.image, use_dynamic_params=True
            )
            self.dynamic_signature = extract_signature_area(self.image, self.dynamic_mask)
            
            self.status_var.set("Processing image with static parameters...")
            self.root.update()
            
            # Process with static parameters
            self.static_mask, self.static_params, static_pixels = extract_signatures(
                self.image, use_dynamic_params=False
            )
            self.static_signature = extract_signature_area(self.image, self.static_mask)
            
            # Display results
            self.display_results()
            self.save_button.config(state=tk.NORMAL)
            self.status_var.set("Signature extraction complete")
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
            self.status_var.set("Error during processing")
            # Print detailed error for debugging
            import traceback
            traceback.print_exc()
    
    def display_results(self):
        # Calculate display size based on available space
        tab_width = self.tab1.winfo_width() or 800
        tab_height = self.tab1.winfo_height() or 400
        
        display_width = int(tab_width / 2) - 40
        display_height = tab_height - 80
        
        # Display dynamic results
        if self.dynamic_mask is not None:
            dynamic_mask_img = Image.fromarray(self.dynamic_mask)
            dynamic_mask_resized = self.resize_preserve_aspect(dynamic_mask_img, display_width, display_height)
            dynamic_mask_photo = ImageTk.PhotoImage(dynamic_mask_resized)
            self.dynamic_mask_label.config(image=dynamic_mask_photo)
            self.dynamic_mask_label.image = dynamic_mask_photo
            
            dynamic_sig_img = Image.fromarray(self.dynamic_signature)
            dynamic_sig_resized = self.resize_preserve_aspect(dynamic_sig_img, display_width, display_height)
            dynamic_sig_photo = ImageTk.PhotoImage(dynamic_sig_resized)
            self.dynamic_signature_label.config(image=dynamic_sig_photo)
            self.dynamic_signature_label.image = dynamic_sig_photo
            
            # Update info text
            self.dynamic_info_label.config(text=f"Dynamic Parameters: {self.format_params(self.dynamic_params)}")
        
        # Display static results
        if self.static_mask is not None:
            static_mask_img = Image.fromarray(self.static_mask)
            static_mask_resized = self.resize_preserve_aspect(static_mask_img, display_width, display_height)
            static_mask_photo = ImageTk.PhotoImage(static_mask_resized)
            self.static_mask_label.config(image=static_mask_photo)
            self.static_mask_label.image = static_mask_photo
            
            static_sig_img = Image.fromarray(self.static_signature)
            static_sig_resized = self.resize_preserve_aspect(static_sig_img, display_width, display_height)
            static_sig_photo = ImageTk.PhotoImage(static_sig_resized)
            self.static_signature_label.config(image=static_sig_photo)
            self.static_signature_label.image = static_sig_photo
            
            # Update info text
            self.static_info_label.config(text=f"Static Parameters: {self.format_params(self.static_params)}")
        
        # Update comparison tab
        self.update_comparison()
    
    def resize_preserve_aspect(self, pil_img, target_width, target_height):
        """Resize image preserving aspect ratio to fit within target dimensions"""
        w, h = pil_img.size
        
        # Calculate ratio to fit within bounds
        ratio = min(target_width / w, target_height / h)
        new_size = (int(w * ratio), int(h * ratio))
        
        return pil_img.resize(new_size, Image.LANCZOS)
    
    def format_params(self, params):
        if not params:
            return "N/A"
        return f"Pixels: {params['signature_pixels']} | P1: {params['constant_parameter_1']} | P2: {params['constant_parameter_2']} | P3: {params['constant_parameter_3']} | P4: {params['constant_parameter_4']}"
    
    def update_comparison(self):
        if self.dynamic_params and self.static_params:
            self.comparison_text.delete(1.0, tk.END)
            
            self.comparison_text.insert(tk.END, "SIGNATURE EXTRACTION COMPARISON\n")
            self.comparison_text.insert(tk.END, "=" * 50 + "\n\n")
            
            self.comparison_text.insert(tk.END, "DYNAMIC PARAMETERS:\n")
            for key, value in self.dynamic_params.items():
                self.comparison_text.insert(tk.END, f"  {key}: {value}\n")
            
            self.comparison_text.insert(tk.END, "\nSTATIC PARAMETERS:\n")
            for key, value in self.static_params.items():
                self.comparison_text.insert(tk.END, f"  {key}: {value}\n")
            
            # Calculate difference
            dyn_pixels = self.dynamic_params.get("signature_pixels", 0)
            static_pixels = self.static_params.get("signature_pixels", 0)
            pixel_diff = abs(dyn_pixels - static_pixels)
            pixel_percent = (pixel_diff / max(dyn_pixels, static_pixels) * 100) if max(dyn_pixels, static_pixels) > 0 else 0
            
            self.comparison_text.insert(tk.END, "\nRESULTS COMPARISON:\n")
            self.comparison_text.insert(tk.END, f"  Pixel difference: {pixel_diff} pixels ({pixel_percent:.2f}%)\n")
            self.comparison_text.insert(tk.END, f"  Better method: {'Dynamic' if dyn_pixels > static_pixels else 'Static'} parameters extracted more signature pixels\n")
    
    def save_signatures(self):
        if self.dynamic_signature is None or self.static_signature is None:
            messagebox.showerror("Error", "No signatures to save")
            return
            
        try:
            save_dir = filedialog.askdirectory(title="Select directory to save signatures")
            if not save_dir:
                return
                
            # Save dynamic signature
            dynamic_path = os.path.join(save_dir, "dynamic_signature.png")
            cv2.imwrite(dynamic_path, cv2.cvtColor(self.dynamic_signature, cv2.COLOR_RGB2BGR))
            
            # Save static signature
            static_path = os.path.join(save_dir, "static_signature.png")
            cv2.imwrite(static_path, cv2.cvtColor(self.static_signature, cv2.COLOR_RGB2BGR))
            
            messagebox.showinfo("Success", f"Signatures saved to:\n{save_dir}")
            self.status_var.set(f"Signatures saved to {save_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save signatures: {str(e)}")
            self.status_var.set("Error saving signatures")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureExtractorApp(root)
    root.mainloop()