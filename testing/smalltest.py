import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SignatureExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Extraction Tuner")
        self.root.geometry("1200x800")
        
        # Default parameters
        self.constant_parameter_1 = 84
        self.constant_parameter_2 = 250
        self.constant_parameter_3 = 100
        self.constant_parameter_4 = 18
        self.threshold_value = 127
        
        # Variables for storing images
        self.original_img = None
        self.processed_img = None
        self.temp_dir = "temp"
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Create UI components
        self.create_ui()
        
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for image display
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load image button
        load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(fill=tk.X, pady=5)
        
        # Save result button
        save_btn = tk.Button(control_frame, text="Save Result", command=self.save_result)
        save_btn.pack(fill=tk.X, pady=5)
        
        # Parameter sliders
        tk.Label(control_frame, text="Threshold Value").pack(anchor=tk.W)
        self.threshold_slider = self.create_slider(control_frame, 0, 255, self.threshold_value, 1)
        
        tk.Label(control_frame, text="Parameter 1 (Small Object Size Ratio)").pack(anchor=tk.W)
        self.param1_slider = self.create_slider(control_frame, 10, 200, self.constant_parameter_1, 1)
        
        tk.Label(control_frame, text="Parameter 2 (Size Multiplier)").pack(anchor=tk.W)
        self.param2_slider = self.create_slider(control_frame, 50, 500, self.constant_parameter_2, 5)
        
        tk.Label(control_frame, text="Parameter 3 (Size Offset)").pack(anchor=tk.W)
        self.param3_slider = self.create_slider(control_frame, 0, 500, self.constant_parameter_3, 5)
        
        tk.Label(control_frame, text="Parameter 4 (Big Size Ratio)").pack(anchor=tk.W)
        self.param4_slider = self.create_slider(control_frame, 2, 50, self.constant_parameter_4, 1)
        
        # Process button
        process_btn = tk.Button(control_frame, text="Process Image", command=self.process_image)
        process_btn.pack(fill=tk.X, pady=20)
        
        # Parameter value display
        self.param_text = tk.Text(control_frame, height=8, width=35)
        self.param_text.pack(fill=tk.X, pady=10)
        self.update_param_text()
        
        # Image display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial setup of plots
        self.ax1.set_title("Original Image")
        self.ax2.set_title("Extracted Signature")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def create_slider(self, parent, min_val, max_val, default_val, resolution):
        var = tk.IntVar(value=default_val)
        slider = tk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                         variable=var, resolution=resolution,
                         command=lambda _: self.update_param_text())
        slider.pack(fill=tk.X)
        return var
    
    def update_param_text(self):
        self.constant_parameter_1 = self.param1_slider.get()
        self.constant_parameter_2 = self.param2_slider.get()
        self.constant_parameter_3 = self.param3_slider.get()
        self.constant_parameter_4 = self.param4_slider.get()
        self.threshold_value = self.threshold_slider.get()
        
        # Calculate derived parameters
        if hasattr(self, 'average') and hasattr(self, 'the_biggest_component'):
            a4_small = ((self.average/self.constant_parameter_1)*self.constant_parameter_2)+self.constant_parameter_3
            a4_big = a4_small * self.constant_parameter_4
            
            text = f"Threshold: {self.threshold_value}\n"
            text += f"Parameter 1: {self.constant_parameter_1}\n"
            text += f"Parameter 2: {self.constant_parameter_2}\n"
            text += f"Parameter 3: {self.constant_parameter_3}\n"
            text += f"Parameter 4: {self.constant_parameter_4}\n"
            text += f"Average Comp Size: {self.average:.2f}\n"
            text += f"Small Object Threshold: {a4_small:.2f}\n"
            text += f"Big Object Threshold: {a4_big:.2f}"
        else:
            text = "Load an image first to see calculated parameters"
            
        self.param_text.delete(1.0, tk.END)
        self.param_text.insert(tk.END, text)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.original_img = cv2.imread(file_path, 0)  # Read as grayscale
            self.display_images(self.original_img, None)
            
            # Pre-process for initial display
            self.process_image()
    
    def save_result(self):
        if self.processed_img is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                   filetypes=[("PNG files", "*.png"), 
                                                             ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.processed_img)
                print(f"Saved result to {file_path}")
    
    def process_image(self):
        if self.original_img is None:
            return
        
        # Get current parameter values
        self.constant_parameter_1 = self.param1_slider.get()
        self.constant_parameter_2 = self.param2_slider.get()
        self.constant_parameter_3 = self.param3_slider.get()
        self.constant_parameter_4 = self.param4_slider.get()
        self.threshold_value = self.threshold_slider.get()
        
        # Apply threshold
        _, binary_img = cv2.threshold(self.original_img, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Connected component analysis
        blobs = binary_img > binary_img.mean()
        blobs_labels = measure.label(blobs, background=1)
        
        # Region analysis
        self.the_biggest_component = 0
        total_area = 0
        counter = 0
        
        for region in regionprops(blobs_labels):
            if region.area > 10:
                total_area += region.area
                counter += 1
            
            if region.area >= 250:
                if region.area > self.the_biggest_component:
                    self.the_biggest_component = region.area
        
        if counter > 0:
            self.average = total_area / counter
        else:
            self.average = 0
        
        # Calculate thresholds
        a4_small_size_outliar_constant = ((self.average/self.constant_parameter_1)*self.constant_parameter_2)+self.constant_parameter_3
        a4_big_size_outliar_constant = a4_small_size_outliar_constant*self.constant_parameter_4
        
        # Remove small objects
        try:
            pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
            
            # Remove large objects
            component_sizes = np.bincount(pre_version.ravel())
            if len(component_sizes) > 1:  # Make sure there are components to filter
                too_small = component_sizes > (a4_big_size_outliar_constant)
                too_small_mask = too_small[pre_version]
                pre_version[too_small_mask] = 0
            
            # Save temporary image
            temp_file = os.path.join(self.temp_dir, "pre_version.png")
            plt.imsave(temp_file, pre_version)
            
            # Read back and apply final threshold
            pre_img = cv2.imread(temp_file, 0)
            _, self.processed_img = cv2.threshold(pre_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Update display
            self.display_images(self.original_img, self.processed_img)
            
            # Update parameter text
            self.update_param_text()
            
        except Exception as e:
            print(f"Error processing image: {e}")
    
    def display_images(self, img1, img2):
        self.ax1.clear()
        self.ax2.clear()
        
        if img1 is not None:
            self.ax1.imshow(img1, cmap='gray')
        self.ax1.set_title("Original Image")
        self.ax1.axis('off')
        
        if img2 is not None:
            self.ax2.imshow(img2, cmap='gray')
        self.ax2.set_title("Extracted Signature")
        self.ax2.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureExtractor(root)
    root.mainloop()