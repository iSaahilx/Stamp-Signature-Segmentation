import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import LassoSelector
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import json
from datetime import datetime
import threading
import queue
import shutil

class SignatureSelectionTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Signature Selection Tool")
        self.root.geometry("1400x900")
        
        # Dataset storage
        self.dataset = []
        self.current_image_path = None
        self.current_image_index = 0
        self.image_paths = []
        self.dataset_dir = "signature_dataset"
        self.dataset_file = os.path.join(self.dataset_dir, "signature_annotations.json")
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            os.makedirs(os.path.join(self.dataset_dir, "images"))
            os.makedirs(os.path.join(self.dataset_dir, "masks"))
            os.makedirs(os.path.join(self.dataset_dir, "results"))
        
        # Variables for storing images
        self.original_img = None
        self.processed_img = None
        self.selection_mask = None
        self.final_mask = None
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
        
        # Selection mode flag
        self.selection_mode = "add"  # "add" or "erase"
        
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
        
        # Navigation tab
        nav_tab = ttk.Frame(notebook)
        notebook.add(nav_tab, text="Navigation")
        
        # Dataset tab
        dataset_tab = ttk.Frame(notebook)
        notebook.add(dataset_tab, text="Dataset")
        
        # ===== Navigation Tab Contents =====
        
        # Selection mode frame
        selection_frame = tk.LabelFrame(nav_tab, text="Selection Mode")
        selection_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Selection mode radio buttons
        self.selection_mode_var = tk.StringVar(value="add")
        tk.Radiobutton(selection_frame, text="Add to Selection (A)", variable=self.selection_mode_var, 
                      value="add", command=self.update_selection_mode).pack(anchor=tk.W, padx=5, pady=2)
        tk.Radiobutton(selection_frame, text="Erase from Selection (E)", variable=self.selection_mode_var, 
                      value="erase", command=self.update_selection_mode).pack(anchor=tk.W, padx=5, pady=2)
        
        # Brush size slider
        brush_frame = tk.Frame(nav_tab)
        brush_frame.pack(fill=tk.X, pady=5)
        tk.Label(brush_frame, text="Brush Size:").pack(side=tk.LEFT)
        self.brush_size_var = tk.IntVar(value=10)
        brush_slider = tk.Scale(brush_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                               variable=self.brush_size_var)
        brush_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Operation buttons
        operation_frame = tk.LabelFrame(nav_tab, text="Operations")
        operation_frame.pack(fill=tk.X, pady=10, padx=5)
        
        # Clear selection button
        tk.Button(operation_frame, text="Clear Selection (C)", 
                 command=self.clear_selection, bg="#ffe0e0").pack(fill=tk.X, pady=5, padx=5)
        
        # Threshold slider for automated processing
        threshold_frame = tk.Frame(nav_tab)
        threshold_frame.pack(fill=tk.X, pady=5)
        tk.Label(threshold_frame, text="Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.IntVar(value=127)
        threshold_slider = tk.Scale(threshold_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
                                  variable=self.threshold_var)
        threshold_slider.pack(fill=tk.X)
        
        # Auto-process button
        tk.Button(nav_tab, text="Auto-Process Image (F5)", 
                 command=self.process_image, bg="#e0e0ff").pack(fill=tk.X, pady=5)
        
        # Apply mask button
        tk.Button(nav_tab, text="Apply Mask (F6)", 
                 command=self.apply_mask, bg="#e0ffe0").pack(fill=tk.X, pady=5)
        
        # Save button
        tk.Button(nav_tab, text="Save to Dataset (F2)", 
                 command=self.save_to_dataset, bg="#e0ffe0").pack(fill=tk.X, pady=5)
        
        # Image navigation frame
        nav_frame = tk.Frame(nav_tab)
        nav_frame.pack(fill=tk.X, pady=10)
        
        # Image navigation buttons
        tk.Button(nav_frame, text="< Prev (Left)", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next (Right) >", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Current image counter
        self.image_counter_var = tk.StringVar(value="Image: 0/0")
        tk.Label(nav_frame, textvariable=self.image_counter_var).pack(side=tk.LEFT, padx=10)
        
        # Load buttons group
        load_frame = tk.Frame(nav_tab)
        load_frame.pack(fill=tk.X, pady=5)
        
        # Load image button
        load_btn = tk.Button(load_frame, text="Load Image (O)", command=self.load_image)
        load_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Load folder button
        load_folder_btn = tk.Button(load_frame, text="Load Folder (F)", command=self.load_folder)
        load_folder_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
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
        
        # ===== Image Display Section =====
        
        # Image display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial setup of plots
        self.ax1.set_title("Original Image & Selection")
        self.ax2.set_title("Processed Signature")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.fig.tight_layout()
        
        # Set up the interactive functionality
        self.setup_brush_interface()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_brush_interface(self):
        self.canvas.draw()
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def setup_key_bindings(self):
        # Navigation keys
        self.root.bind("<Left>", lambda event: self.prev_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        
        # Process and save keys
        self.root.bind("<F5>", lambda event: self.process_image())
        self.root.bind("<F6>", lambda event: self.apply_mask())
        self.root.bind("<F2>", lambda event: self.save_to_dataset())
        
        # Load keys
        self.root.bind("<o>", lambda event: self.load_image())
        self.root.bind("<O>", lambda event: self.load_image())
        self.root.bind("<f>", lambda event: self.load_folder())
        self.root.bind("<F>", lambda event: self.load_folder())
        
        # Selection mode keys
        self.root.bind("<a>", lambda event: self.set_selection_mode("add"))
        self.root.bind("<A>", lambda event: self.set_selection_mode("add"))
        self.root.bind("<e>", lambda event: self.set_selection_mode("erase"))
        self.root.bind("<E>", lambda event: self.set_selection_mode("erase"))
        self.root.bind("<c>", lambda event: self.clear_selection())
        self.root.bind("<C>", lambda event: self.clear_selection())
    
    def set_selection_mode(self, mode):
        self.selection_mode_var.set(mode)
        self.update_selection_mode()
    
    def update_selection_mode(self):
        self.selection_mode = self.selection_mode_var.get()
        self.status_var.set(f"Selection mode: {'Add' if self.selection_mode == 'add' else 'Erase'}")
    
    def on_press(self, event):
        if event.inaxes == self.ax1:
            self.drawing = True
            self.last_x = event.xdata
            self.last_y = event.ydata
            self.apply_brush(event.xdata, event.ydata)
    
    def on_motion(self, event):
        if self.drawing and event.inaxes == self.ax1 and self.last_x is not None and self.last_y is not None:
            # Draw line from last position to current position
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Interpolate points along the line
                points = self.interpolate_points(self.last_x, self.last_y, x, y)
                for point in points:
                    self.apply_brush(point[0], point[1])
                
                self.last_x, self.last_y = x, y
    
    def on_release(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def interpolate_points(self, x1, y1, x2, y2):
        """Interpolate points along a line to ensure smooth brush strokes"""
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_points = max(int(distance * 2), 2)  # At least 2 points
        
        t = np.linspace(0, 1, num_points)
        points = []
        
        for i in range(num_points):
            x = x1 + (x2 - x1) * t[i]
            y = y1 + (y2 - y1) * t[i]
            points.append((x, y))
        
        return points
    
    def apply_brush(self, x, y):
        if self.selection_mask is None or self.original_img is None:
            return
        
        # Convert data coordinates to pixel coordinates
        height, width = self.selection_mask.shape[:2]
        pixel_x = int(round(x))
        pixel_y = int(round(y))
        
        # Get brush size
        brush_size = self.brush_size_var.get()
        
        # Create a temporary mask for this brush stroke
        temp_mask = np.zeros_like(self.selection_mask)
        cv2.circle(temp_mask, (pixel_x, pixel_y), brush_size, 255, -1)
        
        # Apply to selection mask
        if self.selection_mode == "add":
            self.selection_mask = cv2.bitwise_or(self.selection_mask, temp_mask)
        else:  # erase
            cv2.circle(self.selection_mask, (pixel_x, pixel_y), brush_size, 0, -1)
        
        # Update the display
        self.update_display()
    
    def clear_selection(self):
        if self.original_img is not None:
            # Reset selection mask
            height, width = self.original_img.shape[:2]
            self.selection_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Update display
            self.update_display()
            
            self.status_var.set("Selection cleared")
    
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
        self.original_img = cv2.imread(self.current_image_path)
        
        # Convert to grayscale for processing but keep color for display
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        
        # Create an empty selection mask
        height, width = self.original_img.shape[:2]
        self.selection_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Update image counter
        self.image_counter_var.set(f"Image: {self.current_image_index + 1}/{len(self.image_paths)}")
        
        # Display the image
        self.update_display()
        
        # Process the image automatically
        self.process_image()
        
        self.status_var.set(f"Loaded image: {os.path.basename(self.current_image_path)}")
    
    def next_image(self):
        if not self.image_paths:
            return
        
        # Save current image before moving on, if user wants to
        if self.final_mask is not None:
            if messagebox.askyesno("Save", "Save current annotation before moving to next image?"):
                self.save_to_dataset()
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_current_image()
    
    def prev_image(self):
        if not self.image_paths:
            return
        
        # Save current image before moving on, if user wants to
        if self.final_mask is not None:
            if messagebox.askyesno("Save", "Save current annotation before moving to previous image?"):
                self.save_to_dataset()
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.load_current_image()
    
    def process_image(self):
        if self.original_img is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return False
        
        # Apply threshold to grayscale image
        _, self.processed_img = cv2.threshold(self.gray_img, self.threshold_var.get(), 255, cv2.THRESH_BINARY_INV)
        
        # Update display
        self.update_display()
        
        self.status_var.set("Image processed successfully")
        return True
    
    def apply_mask(self):
        if self.original_img is None or self.processed_img is None:
            messagebox.showwarning("Warning", "Please load and process an image first")
            return False
        
        if self.selection_mask is None or np.max(self.selection_mask) == 0:
            messagebox.showwarning("Warning", "Please make a selection first")
            return False
        
        # Apply the selection mask to the processed image
        self.final_mask = cv2.bitwise_and(self.processed_img, self.selection_mask)
        
        # Update the right side display
        self.ax2.clear()
        self.ax2.imshow(self.final_mask, cmap='gray')
        self.ax2.set_title("Final Signature Mask")
        self.ax2.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.status_var.set("Mask applied successfully")
        return True
    
    def update_display(self):
        if self.original_img is None:
            return
        
        # Clear the axes
        self.ax1.clear()
        
        # Display the original image
        rgb_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.ax1.imshow(rgb_img)
        
        # Overlay the selection mask if it exists
        if self.selection_mask is not None and np.max(self.selection_mask) > 0:
            # Create a colored overlay for the mask
            mask_overlay = np.zeros_like(self.original_img)
            mask_overlay[:, :, 0] = 0       # Blue channel
            mask_overlay[:, :, 1] = 0       # Green channel
            mask_overlay[:, :, 2] = 255     # Red channel
            
            # Apply the mask to the overlay
            mask_rgb = cv2.cvtColor(self.selection_mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.bitwise_and(mask_overlay, mask_rgb)
            
            # Display the overlay with transparency
            self.ax1.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), alpha=0.5)
        
        self.ax1.set_title("Original Image & Selection")
        self.ax1.axis('off')
        
        # Update the processed image if it exists
        if self.processed_img is not None:
            self.ax2.clear()
            self.ax2.imshow(self.processed_img, cmap='gray')
            self.ax2.set_title("Processed Signature")
            self.ax2.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_to_dataset(self):
        if self.original_img is None or self.final_mask is None:
            if self.processed_img is not None and self.selection_mask is not None:
                # Try to apply the mask first
                self.apply_mask()
            else:
                messagebox.showwarning("Warning", "Please load, process, and apply a mask first")
                return False
        
        if self.current_image_path is None:
            messagebox.showwarning("Warning", "No image loaded")
            return False
        
        # Create a unique entry ID
        entry_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save original and mask images
        original_save_path = os.path.join(self.dataset_dir, "images", f"{entry_id}_original.png")
        mask_save_path = os.path.join(self.dataset_dir, "masks", f"{entry_id}_mask.png")
        result_save_path = os.path.join(self.dataset_dir, "results", f"{entry_id}_result.png")
        
        cv2.imwrite(original_save_path, self.original_img)
        cv2.imwrite(mask_save_path, self.selection_mask)
        cv2.imwrite(result_save_path, self.final_mask)
        
        # Save annotation information
        entry = {
            "id": entry_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_image": os.path.basename(self.current_image_path),
            "saved_original": os.path.basename(original_save_path),
            "saved_mask": os.path.basename(mask_save_path),
            "saved_result": os.path.basename(result_save_path),
            "threshold": self.threshold_var.get(),
        }
        
        # Add to dataset
        self.dataset.append(entry)
        
        # Save dataset to file
        self.save_dataset()
        
        # Update UI
        self.update_dataset_listbox()
        self.update_stats_label()
        
        self.status_var.set(f"Added entry to dataset: {entry_id}")
        messagebox.showinfo("Success", "Annotation saved to dataset")
        
        return True
    
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
        
        # Load images
        original_path = os.path.join(self.dataset_dir, "images", entry['saved_original'])
        mask_path = os.path.join(self.dataset_dir, "masks", entry['saved_mask'])
        result_path = os.path.join(self.dataset_dir, "results", entry['saved_result'])
        
        if os.path.exists(original_path) and os.path.exists(mask_path) and os.path.exists(result_path):
            self.original_img = cv2.imread(original_path)
            self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
            self.selection_mask = cv2.imread(mask_path, 0)
            self.final_mask = cv2.imread(result_path, 0)
            self.current_image_path = original_path
            
            # Set threshold value
            if 'threshold' in entry:
                self.threshold_var.set(entry['threshold'])
            
            # Update display
            self.update_display()
            
            # Update right pane with final mask
            self.ax2.clear()
            self.ax2.imshow(self.final_mask, cmap='gray')
            self.ax2.set_title("Final Signature Mask")
            self.ax2.axis('off')
            self.fig.tight_layout()
            self.canvas.draw()
            
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
        mask_path = os.path.join(self.dataset_dir, "masks", entry['saved_mask'])
        result_path = os.path.join(self.dataset_dir, "results", entry['saved_result'])
        
        try:
            if os.path.exists(original_path):
                os.remove(original_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)
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
            
            # Create a temporary directory for export
            temp_export_dir = os.path.join(self.temp_dir, "export")
            if os.path.exists(temp_export_dir):
                shutil.rmtree(temp_export_dir)
            os.makedirs(temp_export_dir)
            
            # Copy dataset files
            shutil.copy(self.dataset_file, temp_export_dir)
            shutil.copytree(os.path.join(self.dataset_dir, "images"), os.path.join(temp_export_dir, "images"))
            shutil.copytree(os.path.join(self.dataset_dir, "masks"), os.path.join(temp_export_dir, "masks"))
            shutil.copytree(os.path.join(self.dataset_dir, "results"), os.path.join(temp_export_dir, "results"))
            
            # Create zip file
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_export_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up temporary export directory
            shutil.rmtree(temp_export_dir)
            
            self.status_var.set(f"Dataset exported successfully to {export_path}")
            messagebox.showinfo("Success", "Dataset exported successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export dataset: {str(e)}")
            self.status_var.set("Dataset export failed")
if __name__ == "__main__":
    root = tk.Tk()
    app = SignatureSelectionTool(root)
    root.mainloop()