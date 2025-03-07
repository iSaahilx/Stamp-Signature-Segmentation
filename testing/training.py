import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue

class SignatureDatasetCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Dataset Creator")
        self.root.geometry("1400x900")
        
        # Default parameters
        self.constant_parameter_1 = 84
        self.constant_parameter_2 = 250
        self.constant_parameter_3 = 100
        self.constant_parameter_4 = 18
        self.threshold_value = 127
        
        # Dataset storage
        self.dataset = []
        self.current_image_path = None
        self.current_image_index = 0
        self.image_paths = []
        self.dataset_dir = "signature_dataset"
        self.dataset_file = os.path.join(self.dataset_dir, "signature_parameters.json")
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            os.makedirs(os.path.join(self.dataset_dir, "images"))
            os.makedirs(os.path.join(self.dataset_dir, "results"))
        
        # Variables for storing images
        self.original_img = None
        self.processed_img = None
        self.temp_dir = "temp"
        
        # Batch processing queue and flag
        self.processing_queue = queue.Queue()
        self.is_batch_processing = False
        
        # Create temp directory if it doesn't exist
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        
        # Load existing dataset if it exists
        self.load_dataset()
        
        # Create UI components
        self.create_ui()
        
        # Set up key bindings
        self.setup_key_bindings()
        
    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = tk.Frame(main_frame, width=350)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Right panel for image display
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add notebook for tabs
        notebook = ttk.Notebook(control_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Parameter tab
        param_tab = ttk.Frame(notebook)
        notebook.add(param_tab, text="Parameters")
        
        # Dataset tab
        dataset_tab = ttk.Frame(notebook)
        notebook.add(dataset_tab, text="Dataset")
        
        # Batch Processing tab
        batch_tab = ttk.Frame(notebook)
        notebook.add(batch_tab, text="Batch Processing")
        
        # ===== Parameter Tab Contents =====
        
        # Parameter sliders with fine and coarse adjustments
        self.create_parameter_controls(param_tab)
        
        # Image navigation frame
        nav_frame = tk.Frame(param_tab)
        nav_frame.pack(fill=tk.X, pady=5)
        
        # Image navigation buttons
        tk.Button(nav_frame, text="< Prev (Left)", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next (Right) >", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Current image counter
        self.image_counter_var = tk.StringVar(value="Image: 0/0")
        tk.Label(nav_frame, textvariable=self.image_counter_var).pack(side=tk.LEFT, padx=10)
        
        # Process button with keyboard shortcut info
        process_btn = tk.Button(param_tab, text="Process Image (F5)", command=self.process_image, bg="#e0e0ff")
        process_btn.pack(fill=tk.X, pady=10)
        
        # Save to dataset button with keyboard shortcut info
        save_dataset_btn = tk.Button(param_tab, text="Save to Dataset (F2)", command=self.save_to_dataset, bg="#e0ffe0")
        save_dataset_btn.pack(fill=tk.X, pady=5)
        
        # Load buttons group
        load_frame = tk.Frame(param_tab)
        load_frame.pack(fill=tk.X, pady=5)
        
        # Load image button
        load_btn = tk.Button(load_frame, text="Load Image (O)", command=self.load_image)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Load folder button
        load_folder_btn = tk.Button(load_frame, text="Load Folder (F)", command=self.load_folder)
        load_folder_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Parameter value display
        self.param_text = tk.Text(param_tab, height=10, width=40)
        self.param_text.pack(fill=tk.X, pady=10)
        self.update_param_text()
        
        # ===== Dataset Tab Contents =====
        
        # Dataset statistics
        tk.Label(dataset_tab, text="Dataset Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        self.stats_label = tk.Label(dataset_tab, text="No images in dataset")
        self.stats_label.pack(anchor=tk.W, pady=5)
        
        # Dataset list with scrollbar
        tk.Label(dataset_tab, text="Saved Entries:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        
        list_frame = tk.Frame(dataset_tab)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.dataset_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=15)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.dataset_listbox.yview)
        
        # Dataset buttons
        btn_frame = tk.Frame(dataset_tab)
        btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(btn_frame, text="Load Selected", command=self.load_selected_entry).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Delete Selected", command=self.delete_selected_entry).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export Dataset", command=self.export_dataset).pack(side=tk.LEFT, padx=5)
        
        # Populate dataset listbox
        self.update_dataset_listbox()
        self.update_stats_label()
        
        # ===== Batch Processing Tab =====
        tk.Label(batch_tab, text="Batch Processing:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10,5))
        
        # Batch progress
        self.batch_progress_var = tk.StringVar(value="Ready")
        tk.Label(batch_tab, textvariable=self.batch_progress_var).pack(anchor=tk.W, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(batch_tab, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)
        
        # Batch process options
        options_frame = tk.Frame(batch_tab)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Auto-save checkbox
        self.auto_save_var = tk.BooleanVar(value=True)
        auto_save_chk = tk.Checkbutton(options_frame, text="Auto-save to dataset", variable=self.auto_save_var)
        auto_save_chk.pack(anchor=tk.W)
        
        # Skip existing checkbox
        self.skip_existing_var = tk.BooleanVar(value=True)
        skip_existing_chk = tk.Checkbutton(options_frame, text="Skip already processed images", variable=self.skip_existing_var)
        skip_existing_chk.pack(anchor=tk.W)
        
        # Batch buttons
        batch_btn_frame = tk.Frame(batch_tab)
        batch_btn_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(batch_btn_frame, text="Process All Loaded Images (F8)", 
                 command=self.start_batch_processing, bg="#e0ffe0").pack(fill=tk.X, pady=5)
        
        tk.Button(batch_btn_frame, text="Stop Batch Processing (Esc)", 
                 command=self.stop_batch_processing, bg="#ffe0e0").pack(fill=tk.X, pady=5)
        
        # Batch stats
        self.batch_stats_var = tk.StringVar(value="No batch stats available")
        tk.Label(batch_tab, textvariable=self.batch_stats_var).pack(anchor=tk.W, pady=10)
        
        # ===== Image Display Section =====
        
        # Image display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial setup of plots
        self.ax1.set_title("Original Image")
        self.ax2.set_title("Extracted Signature")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_key_bindings(self):
        # Navigation keys
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        
        # Process and save keys
        self.root.bind("<F5>", lambda event: self.process_image())
        self.root.bind("<F2>", lambda event: self.save_to_dataset())
        
        # Load keys
        self.root.bind("<o>", lambda event: self.load_image())
        self.root.bind("<O>", lambda event: self.load_image())
        self.root.bind("<f>", lambda event: self.load_folder())
        self.root.bind("<F>", lambda event: self.load_folder())
        
        # Batch processing keys
        self.root.bind("<F8>", lambda event: self.start_batch_processing())
        self.root.bind("<Escape>", lambda event: self.stop_batch_processing())
    
    def create_parameter_controls(self, parent):
        # Function to create a parameter control with fine and coarse adjustments
        def create_param_control(label_text, min_val, max_val, default_val, fine_step, coarse_step, var_name):
            frame = tk.Frame(parent)
            frame.pack(fill=tk.X, pady=5)
            
            tk.Label(frame, text=label_text).pack(anchor=tk.W)
            
            # Store variable reference in self
            var = tk.IntVar(value=default_val)
            setattr(self, var_name, var)
            
            # Main slider
            slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, 
                             variable=var, resolution=fine_step,
                             command=lambda _: self.on_slider_change())
            slider.pack(fill=tk.X)
            
            # Buttons for coarse adjustments
            btn_frame = tk.Frame(frame)
            btn_frame.pack(fill=tk.X)
            
            tk.Button(btn_frame, text=f"-{coarse_step}", 
                     command=lambda: self.adjust_parameter(var, -coarse_step, min_val, max_val)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text=f"-{fine_step}", 
                     command=lambda: self.adjust_parameter(var, -fine_step, min_val, max_val)).pack(side=tk.LEFT, padx=2)
            tk.Button(btn_frame, text=f"+{fine_step}", 
                     command=lambda: self.adjust_parameter(var, fine_step, min_val, max_val)).pack(side=tk.RIGHT, padx=2)
            tk.Button(btn_frame, text=f"+{coarse_step}", 
                     command=lambda: self.adjust_parameter(var, coarse_step, min_val, max_val)).pack(side=tk.RIGHT, padx=2)
            
            return var
        
        # Create parameter controls
        tk.Label(parent, text="Parameter Tuning:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0,5))
        
        self.threshold_var = create_param_control(
            "Threshold Value", 0, 255, self.threshold_value, 1, 10, "threshold_var")
        
        self.param1_var = create_param_control(
            "Parameter 1 (Small Object Size Ratio)", 10, 200, self.constant_parameter_1, 1, 5, "param1_var")
        
        self.param2_var = create_param_control(
            "Parameter 2 (Size Multiplier)", 50, 500, self.constant_parameter_2, 5, 25, "param2_var")
        
        self.param3_var = create_param_control(
            "Parameter 3 (Size Offset)", 0, 500, self.constant_parameter_3, 5, 25, "param3_var")
        
        self.param4_var = create_param_control(
            "Parameter 4 (Big Size Ratio)", 2, 50, self.constant_parameter_4, 1, 5, "param4_var")
    
    def adjust_parameter(self, var, amount, min_val, max_val):
        new_val = max(min_val, min(max_val, var.get() + amount))
        var.set(new_val)
        self.on_slider_change()
    
    def on_slider_change(self):
        # Update parameters
        self.update_param_text()
        
        # Auto-process if an image is loaded
        if self.original_img is not None:
            self.process_image()
    
    def update_param_text(self):
        self.constant_parameter_1 = self.param1_var.get()
        self.constant_parameter_2 = self.param2_var.get()
        self.constant_parameter_3 = self.param3_var.get()
        self.constant_parameter_4 = self.param4_var.get()
        self.threshold_value = self.threshold_var.get()
        
        # Calculate derived parameters
        if hasattr(self, 'average') and hasattr(self, 'the_biggest_component'):
            a4_small = ((self.average/self.constant_parameter_1)*self.constant_parameter_2)+self.constant_parameter_3
            a4_big = a4_small * self.constant_parameter_4
            
            text = f"Current Image: {os.path.basename(self.current_image_path) if self.current_image_path else 'None'}\n\n"
            text += f"Threshold: {self.threshold_value}\n"
            text += f"Parameter 1: {self.constant_parameter_1}\n"
            text += f"Parameter 2: {self.constant_parameter_2}\n"
            text += f"Parameter 3: {self.constant_parameter_3}\n"
            text += f"Parameter 4: {self.constant_parameter_4}\n\n"
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
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.load_current_image()
    
    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Get all image files in the folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_paths = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and 
                os.path.splitext(f)[1].lower() in image_extensions
            ]
            
            if not self.image_paths:
                messagebox.showwarning("Warning", "No image files found in the selected folder")
                return
            
            self.image_paths.sort()  # Sort files by name
            self.current_image_index = 0
            self.load_current_image()
            
            self.status_var.set(f"Loaded {len(self.image_paths)} images from folder")
    
    def load_current_image(self):
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            return
        
        self.current_image_path = self.image_paths[self.current_image_index]
        self.original_img = cv2.imread(self.current_image_path, 0)  # Read as grayscale
        
        # Update image counter
        self.image_counter_var.set(f"Image: {self.current_image_index + 1}/{len(self.image_paths)}")
        
        # Display the image
        self.display_images(self.original_img, None)
        
        # Process the image automatically
        self.process_image()
        
        self.status_var.set(f"Loaded image: {os.path.basename(self.current_image_path)}")
    
    def next_image(self):
        if not self.image_paths:
            return
        
        # Optionally auto-save current image before moving on
        if hasattr(self, 'auto_save_var') and self.auto_save_var.get() and self.processed_img is not None:
            self.save_to_dataset()
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_current_image()
    
    def prev_image(self):
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.load_current_image()
    
    def process_image(self):
        if self.original_img is None:
            if not self.is_batch_processing:
                messagebox.showwarning("Warning", "Please load an image first")
            return False
        
        # Get current parameter values
        self.constant_parameter_1 = self.param1_var.get()
        self.constant_parameter_2 = self.param2_var.get()
        self.constant_parameter_3 = self.param3_var.get()
        self.constant_parameter_4 = self.param4_var.get()
        self.threshold_value = self.threshold_var.get()
        
        try:
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
            
            if not self.is_batch_processing:
                self.status_var.set("Image processed successfully")
            
            return True
            
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            if not self.is_batch_processing:
                self.status_var.set(error_msg)
                messagebox.showerror("Processing Error", error_msg)
            return False
    
    def save_to_dataset(self):
        if self.original_img is None or self.processed_img is None:
            if not self.is_batch_processing:
                messagebox.showwarning("Warning", "Please load and process an image first")
            return False
        
        if self.current_image_path is None:
            if not self.is_batch_processing:
                messagebox.showwarning("Warning", "No image loaded")
            return False
        
        # Skip if already exists in dataset
        if self.skip_existing_var.get() and self.is_batch_processing:
            orig_filename = os.path.basename(self.current_image_path)
            for entry in self.dataset:
                if entry['original_image'] == orig_filename:
                    return True  # Skip this one
        
        # Create a unique entry ID
        entry_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original and processed images
        original_save_path = os.path.join(self.dataset_dir, "images", f"{entry_id}_original.png")
        result_save_path = os.path.join(self.dataset_dir, "results", f"{entry_id}_result.png")
        
        cv2.imwrite(original_save_path, self.original_img)
        cv2.imwrite(result_save_path, self.processed_img)
        
        # Save parameter information
        entry = {
            "id": entry_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_image": os.path.basename(self.current_image_path),
            "saved_original": os.path.basename(original_save_path),
            "saved_result": os.path.basename(result_save_path),
            "parameters": {
                "threshold": self.threshold_value,
                "param1": self.constant_parameter_1,
                "param2": self.constant_parameter_2,
                "param3": self.constant_parameter_3,
                "param4": self.constant_parameter_4
            },
            "metrics": {
                "average_component_size": float(self.average),
                "biggest_component": int(self.the_biggest_component)
            }
        }
        
        # Add to dataset
        self.dataset.append(entry)
        
        # Save dataset to file
        self.save_dataset()
        
        # Update UI
        self.update_dataset_listbox()
        self.update_stats_label()
        
        if not self.is_batch_processing:
            self.status_var.set(f"Added entry to dataset: {entry_id}")
            messagebox.showinfo("Success", "Parameters and results saved to dataset")
        
        return True
    
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
    
    def load_dataset(self):
        if os.path.exists(self.dataset_file):
            try:
                with open(self.dataset_file, 'r') as f:
                    self.dataset = json.load(f)
            except:
                self.dataset = []
        else:
            self.dataset = []
    
    def save_dataset(self):
        with open(self.dataset_file, 'w') as f:
            json.dump(self.dataset, f, indent=4)
    
    def update_dataset_listbox(self):
        self.dataset_listbox.delete(0, tk.END)
        for entry in self.dataset:
            display_text = f"{entry['timestamp']} - Img: {entry['original_image']}"
            self.dataset_listbox.insert(tk.END, display_text)
    
    def update_stats_label(self):
        if self.dataset:
            stats_text = f"Total entries: {len(self.dataset)}\n"
            stats_text += f"Last entry: {self.dataset[-1]['timestamp']}"
            self.stats_label.config(text=stats_text)
        else:
            self.stats_label.config(text="No images in dataset")
    
    def load_selected_entry(self):
        selected_idx = self.dataset_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("Warning", "Please select an entry from the dataset")
            return
        
        entry = self.dataset[selected_idx[0]]
        
        # Load parameters
        self.threshold_var.set(entry['parameters']['threshold'])
        self.param1_var.set(entry['parameters']['param1'])
        self.param2_var.set(entry['parameters']['param2'])
        self.param3_var.set(entry['parameters']['param3'])
        self.param4_var.set(entry['parameters']['param4'])
        
        # Load images
        original_path = os.path.join(self.dataset_dir, "images", entry['saved_original'])
        result_path = os.path.join(self.dataset_dir, "results", entry['saved_result'])
        
        if os.path.exists(original_path) and os.path.exists(result_path):
            self.original_img = cv2.imread(original_path, 0)
            self.processed_img = cv2.imread(result_path, 0)
            self.current_image_path = original_path
            
            # Update average and biggest component
            self.average = entry['metrics']['average_component_size']
            self.the_biggest_component = entry['metrics']['biggest_component']
            
            # Display images
            self.display_images(self.original_img, self.processed_img)
            
            # Update parameters text
            self.update_param_text()
            
            self.status_var.set(f"Loaded entry from dataset: {entry['id']}")
        else:
            messagebox.showerror("Error", "Could not load saved images")
    
    def delete_selected_entry(self):
        selected_idx = self.dataset_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("Warning", "Please select an entry from the dataset")
            return
        
        entry = self.dataset[selected_idx[0]]
        
        # Confirm delete
        if not messagebox.askyesno("Confirm Delete", f"Delete entry from {entry['timestamp']}?"):
            return
        
        # Remove files
        original_path = os.path.join(self.dataset_dir, "images", entry['saved_original'])
        result_path = os.path.join(self.dataset_dir, "results", entry['saved_result'])
        
        try:
            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(result_path):
                os.remove(result_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete files: {str(e)}")
            return
        
        # Remove entry from dataset
        del self.dataset[selected_idx[0]]
        
        # Save updated dataset
        self.save_dataset()
        
        # Update UI
        self.update_dataset_listbox()
        self.update_stats_label()
        
        self.status_var.set(f"Deleted entry: {entry['id']}")
    
    def export_dataset(self):
        export_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            title="Export Dataset"
        )
        
        if not export_path:
            return
        
        try:
            import zipfile
            import shutil
            
            # Create a temporary directory for export
            temp_export_dir = os.path.join(self.temp_dir, "export")
            if os.path.exists(temp_export_dir):
                shutil.rmtree(temp_export_dir)
            os.makedirs(temp_export_dir)
            
            # Copy dataset files
            shutil.copy(self.dataset_file, temp_export_dir)
            shutil.copytree(os.path.join(self.dataset_dir, "images"), os.path.join(temp_export_dir, "images"))
            shutil.copytree(os.path.join(self.dataset_dir, "results"), os.path.join(temp_export_dir, "results"))
            
            # Create zip file
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_export_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up
            shutil.rmtree(temp_export_dir)
            
            self.status_var.set(f"Dataset exported to {export_path}")
            messagebox.showinfo("Success", "Dataset exported successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Could not export dataset: {str(e)}")
    
    def start_batch_processing(self):
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please load images first")
            return
        
        # Clear queue and add all images
        self.processing_queue.queue.clear()
        for path in self.image_paths:
            self.processing_queue.put(path)
        
        # Set batch processing flag
        self.is_batch_processing = True
        
        # Start processing thread
        self.batch_thread = threading.Thread(target=self.process_batch)
        self.batch_thread.start()
        
        # Update UI
        self.batch_progress_var.set("Processing...")
        self.progress_var.set(0)
        self.status_var.set("Batch processing started")
    
    def process_batch(self):
        total_images = self.processing_queue.qsize()
        processed_count = 0
        success_count = 0
        
        while not self.processing_queue.empty() and self.is_batch_processing:
            image_path = self.processing_queue.get()
            self.current_image_path = image_path
            self.original_img = cv2.imread(image_path, 0)
            
            if self.process_image():
                if self.save_to_dataset():
                    success_count += 1
            
            processed_count += 1
            progress = (processed_count / total_images) * 100
            self.progress_var.set(progress)
            self.batch_stats_var.set(f"Processed: {processed_count}/{total_images} | Success: {success_count}")
        
        # Update UI after processing
        self.is_batch_processing = False
        self.batch_progress_var.set("Batch processing complete")
        self.status_var.set(f"Batch processing complete: {success_count}/{total_images} images processed successfully")
    
    def stop_batch_processing(self):
        if self.is_batch_processing:
            self.is_batch_processing = False
            self.batch_progress_var.set("Batch processing stopped")
            self.status_var.set("Batch processing stopped by user")
    
    def on_closing(self):
        if self.is_batch_processing:
            if not messagebox.askokcancel("Quit", "Batch processing is still running. Are you sure you want to quit?"):
                return
        
        # Clean up temp directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except:
            pass
        
        self.root.destroy()

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureDatasetCreator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()