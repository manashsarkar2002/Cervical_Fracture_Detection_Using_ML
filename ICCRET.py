import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset (Assuming images are stored in 'Pic' folder)
def load_images_from_folder(image_path, label):
    images = []
    labels = []
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    if img is not None:
        img = cv2.resize(img, (128, 128))  # Resize for consistency
        images.append(img)
        labels.append(label)
    
    return images, labels

# Load normal and fractured C5 vertebra images
normal_images, normal_labels = load_images_from_folder("Pic/normalc5pic1.jpg", label=0)

fracture_images1, fracture_labels1 = load_images_from_folder("Pic/c5pic1.jpeg", label=1)
fracture_images2, fracture_labels2 = load_images_from_folder("Pic/c5pic2.jpeg", label=1)
fracture_images3, fracture_labels3 = load_images_from_folder("Pic/c5pic3.jpeg", label=1)

# Combine dataset
X = np.array(normal_images + fracture_images1 + fracture_images2 + fracture_images3)
y = np.array(normal_labels + fracture_labels1 + fracture_labels2 + fracture_labels3)

# 2. Apply Sobel Edge Detection
def sobel_features(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # X-axis edge detection
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # Y-axis edge detection
    sobel_combined = cv2.magnitude(sobelx, sobely)  # Magnitude of edges
    
    # Feature extraction (mean, variance, edge count)
    mean_val = np.mean(sobel_combined)
    var_val = np.var(sobel_combined)
    edge_count = np.sum(sobel_combined > 50)  # Threshold to count edges
    
    return [mean_val, var_val, edge_count]

# Extract features
X_features = np.array([sobel_features(img) for img in X])

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 5. Predictions & Evaluation
y_pred = rf_classifier.predict(X_test)

# Accuracy & Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Visualize Sobel Edge Detection Example
sample_img = X[0]
sobelx = cv2.Sobel(sample_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(sample_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1), plt.imshow(sample_img, cmap='gray'), plt.title("Original C5 Image")
plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray'), plt.title("Sobel X")
plt.subplot(1, 3, 3), plt.imshow(sobel_combined, cmap='gray'), plt.title("Sobel Edge Detection")
plt.show()


