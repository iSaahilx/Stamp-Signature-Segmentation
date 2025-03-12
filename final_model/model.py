import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import io
import json
import base64

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    messagebox.showerror("Missing Package", "Please install the inference_sdk package with: pip install inference_sdk")
    exit(1)

class ImageSegmentationApp:
    def _init_(self, root):
        self.root = root
        self.root.title("Image Segmentation GUI")
        self.root.geometry("900x600")
        self.client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="[Sensitive data removed]")
        self.original_image = None
        self.processed_images = []
        self.current_processed_index = 0
        self.create_ui()

    def create_ui(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        select_btn = tk.Button(button_frame, text="Select Image", command=self.select_image)
        select_btn.pack(side=tk.LEFT, padx=5)
        save_btn = tk.Button(button_frame, text="Save Processed Image", command=self.save_image)
        save_btn.pack(side=tk.LEFT, padx=5)
        self.nav_frame = tk.Frame(button_frame)
        self.nav_frame.pack(side=tk.LEFT, padx=20)
        self.prev_btn = tk.Button(self.nav_frame, text="< Prev", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.next_btn = tk.Button(self.nav_frame, text="Next >", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        self.image_counter = tk.Label(self.nav_frame, text="Image: 0/0")
        self.image_counter.pack(side=tk.LEFT, padx=5)
        self.prev_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)
        self.status_label = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        image_frame = tk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = tk.Frame(image_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left_frame, text="Original Image").pack()
        self.left_canvas = tk.Canvas(left_frame, width=400, height=400, bg="lightgray")
        self.left_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        right_frame = tk.Frame(image_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_label = tk.Label(right_frame, text="Processed Image")
        self.right_label.pack()
        self.right_canvas = tk.Canvas(right_frame, width=400, height=400, bg="lightgray")
        self.right_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=100, mode="indeterminate")

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.jpeg;.png;.bmp")])
        if not file_path:
            return
        self.status_label.config(text=f"Loading image: {os.path.basename(file_path)}")
        self.root.update()
        try:
            self.processed_images = []
            self.current_processed_index = 0
            self.progress.pack(before=self.status_label, fill=tk.X, padx=10, pady=5)
            self.progress.start()
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not read the image file")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.original_image = Image.fromarray(image)
            self.show_image(self.original_image, self.left_canvas)
            self.status_label.config(text="Processing image with Roboflow API...")
            self.root.update()
            result = self.client.run_workflow(workspace_name="[Sensitive data removed]", workflow_id="[Sensitive data removed]", images={"image": file_path}, use_cache=True)
            print("API Result type:", type(result))
            if isinstance(result, list) and len(result) > 0:
                print("First item keys:", result[0].keys() if isinstance(result[0], dict) else "Not a dict")
            extracted_images = self.extract_images_from_result(result)
            if extracted_images:
                self.processed_images = extracted_images
                self.update_image_counter()
                self.show_current_processed_image()
                self.status_label.config(text=f"Successfully processed. Found {len(self.processed_images)} output images.")
            else:
                self.status_label.config(text="No processed images found in the API response.")
            self.update_navigation_buttons()
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            print(f"Exception details: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.progress.stop()
            self.progress.pack_forget()

    def extract_images_from_result(self, result):
        extracted_images = []
        try:
            if isinstance(result, dict):
                for key, value in result.items():
                    if key.endswith('_image') and isinstance(value, str):
                        if value.startswith('http'):
                            extracted_images.append(('URL', key, value))
                        elif value.startswith('data:image') or self.is_base64(value):
                            extracted_images.append(('Base64', key, value))
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, str):
                                if value.startswith('http') and ('image' in key.lower() or key == 'url'):
                                    extracted_images.append(('URL', key, value))
                                elif value.startswith('data:image') or self.is_base64(value):
                                    extracted_images.append(('Base64', key, value))
                            elif isinstance(value, dict):
                                if 'visualization' in value:
                                    viz = value['visualization']
                                    if isinstance(viz, str):
                                        if viz.startswith('http'):
                                            extracted_images.append(('URL', 'visualization', viz))
                                        elif viz.startswith('data:image') or self.is_base64(viz):
                                            extracted_images.append(('Base64', 'visualization', viz))
                                for subkey, subvalue in value.items():
                                    if isinstance(subvalue, str) and (subvalue.startswith('http') or subvalue.startswith('data:image') or self.is_base64(subvalue)):
                                        extracted_images.append(('URL' if subvalue.startswith('http') else 'Base64', f"{key}.{subkey}", subvalue))
            loaded_images = []
            for img_type, img_name, img_data in extracted_images:
                print(f"Found {img_type} image: {img_name}")
                try:
                    if img_type == 'URL':
                        response = requests.get(img_data)
                        if response.status_code == 200:
                            image_np = np.asarray(bytearray(response.content), dtype=np.uint8)
                            image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
                            if image is not None:
                                if len(image.shape) == 3 and image.shape[2] == 4:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                                else:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                loaded_images.append((img_name, Image.fromarray(image)))
                    elif img_type == 'Base64':
                        if img_data.startswith('data:image'):
                            img_data = img_data.split(',')[1]
                        image_data = base64.b64decode(img_data)
                        image_np = np.frombuffer(image_data, dtype=np.uint8)
                        image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
                        if image is not None:
                            if len(image.shape) == 3 and image.shape[2] == 4:
                                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                            else:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            loaded_images.append((img_name, Image.fromarray(image)))
                except Exception as e:
                    print(f"Error loading image {img_name}: {str(e)}")
            return loaded_images
        except Exception as e:
            print(f"Error extracting images: {str(e)}")
            return []

    def is_base64(self, s):
        try:
            if not all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                return False
            base64.b64decode(s)
            return True
        except:
            return False

    def show_current_processed_image(self):
        if not self.processed_images:
            return
        if 0 <= self.current_processed_index < len(self.processed_images):
            img_name, img = self.processed_images[self.current_processed_index]
            self.right_label.config(text=f"Processed Image: {img_name}")
            self.show_image(img, self.right_canvas)

    def prev_image(self):
        if self.processed_images and self.current_processed_index > 0:
            self.current_processed_index -= 1
            self.show_current_processed_image()
            self.update_image_counter()
            self.update_navigation_buttons()

    def next_image(self):
        if self.processed_images and self.current_processed_index < len(self.processed_images) - 1:
            self.current_processed_index += 1
            self.show_current_processed_image()
            self.update_image_counter()
            self.update_navigation_buttons()

    def update_image_counter(self):
        total = len(self.processed_images)
        current = self.current_processed_index + 1 if total > 0 else 0
        self.image_counter.config(text=f"Image: {current}/{total}")

    def update_navigation_buttons(self):
        if not self.processed_images:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            return
        self.prev_btn.config(state=tk.NORMAL if self.current_processed_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_processed_index < len(self.processed_images) - 1 else tk.DISABLED)

    def show_image(self, img, canvas):
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1:
            canvas_width = 400
        if canvas_height <= 1:
            canvas_height = 400
        img_copy = img.copy()
        img_width, img_height = img_copy.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        img_resized = img_copy.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

    def save_image(self):
        if not self.processed_images:
            messagebox.showinfo("Info", "No processed image to save")
            return
        current_image = self.processed_images[self.current_processed_index][1]
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", ".png"), ("JPEG files", ".jpg"), ("All files", ".")])
        if not file_path:
            return
        try:
            current_image.save(file_path)
            self.status_label.config(text=f"Image saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            self.status_label.config(text=f"Error saving: {str(e)}")

if _name_ == "_main_":
    try:
        root = tk.Tk()
        app = ImageSegmentationApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Critical Error", f"Application crashed: {str(e)}")