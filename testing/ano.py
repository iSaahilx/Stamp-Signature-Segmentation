import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from rembg import remove

def preprocess_image(img):
    # Use rembg to remove the background
    output = remove(img)
    return output

def update_image():
    if original_img is None:
        return
    
    # Process the image using rembg
    processed_img = preprocess_image(original_img)
    
    # Convert the output to grayscale for display
    processed_img_gray = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGBA2GRAY)
    
    # Display the original and processed images
    display_images(original_img, processed_img_gray)

def display_images(img1, img2):
    imgs = [img1, img2]
    for i, img in enumerate(imgs):
        img = cv2.resize(img, (300, 300))
        if i == 1:  # Convert grayscale to 3 channels for display
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        image_labels[i].config(image=img)
        image_labels[i].image = img

def load_image():
    global original_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
    if file_path:
        original_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        update_image()

# Initialize UI
top = tk.Tk()
top.title("Background Removal with rembg")

original_img = None
image_labels = [tk.Label(top) for _ in range(2)]
for label in image_labels:
    label.pack(side=tk.LEFT, padx=5, pady=5)

# Load Image Button
tk.Button(top, text="Load Image", command=load_image).pack()

top.mainloop()