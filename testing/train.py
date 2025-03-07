import json
import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load JSON data
def load_data(json_file, image_folder):
    with open(json_file, "r") as f:
        data = json.load(f)

    X_images = []
    Y_parameters = []

    for entry in data:
        img_path = os.path.join(image_folder, entry["original_image"])
        if os.path.exists(img_path):
            X_images.append(img_path)
            Y_parameters.append([
                entry["parameters"]["threshold"],
                entry["parameters"]["param1"],
                entry["parameters"]["param2"],
                entry["parameters"]["param3"],
                entry["parameters"]["param4"]
            ])
        else:
            print(f"Image not found: {img_path}")

    return X_images, np.array(Y_parameters)

# Feature extraction function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    img = cv2.resize(img, (128, 128))  # Standardize size

    # Extract features
    edges = cv2.Canny(img, 50, 150)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate([hog_features, edges.flatten(), thresh.flatten()])

# Load dataset
json_file = "D:/Hackathon/images/dataset/dataset_45/signature_parameters.json"
image_folder = "D:\Hackathon\images\preprocessed without augmentation"
X_images, Y_parameters = load_data(json_file, image_folder)

# Extract features for all images
X_features = []
for img in X_images:
    features = extract_features(img)
    if features is not None:
        X_features.append(features)
    else:
        print(f"Skipping image due to feature extraction failure: {img}")

X_features = np.array(X_features)

# Check if any features were extracted
if X_features.size == 0:
    raise ValueError("No features were extracted from the images. Check the image paths and feature extraction function.")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_parameters, test_size=0.2, random_state=42)

# Train a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and scaler
joblib.dump(model, "signature_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model training completed and saved.")

# Function to predict parameters for a new image
def predict_parameters(image_path):
    model = joblib.load("signature_model.pkl")
    scaler = joblib.load("scaler.pkl")

    features = extract_features(image_path)
    if features is None:
        raise ValueError("Feature extraction failed for the new image.")

    features_scaled = scaler.transform([features])

    predicted_params = model.predict(features_scaled)[0]
    param_names = ["threshold", "param1", "param2", "param3", "param4"]

    return dict(zip(param_names, predicted_params))

# Example usage
new_image = "D:\Hackathon\images\preprocessed without augmentation\preprocessed_signature_427_1.jpg"  # Replace with actual test image path
predicted_params = predict_parameters(new_image)
print("Predicted Parameters:", predicted_params)