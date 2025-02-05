import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset (Single Image Handling)
def load_image(img_path, label):
    if not os.path.exists(img_path):
        print(f"Warning: Image '{img_path}' not found.")
        return None, None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    if img is not None:
        img = cv2.resize(img, (128, 128))  # Resize for consistency
        return img, label

    return None, None

# Load normal and fractured C5 vertebra images
normal_img, normal_label = load_image("Pic/normalc5pic1.jpg", label=0)
fracture_img, fracture_label = load_image("Pic/C5 fracture1.jpeg", label=1)

# Check if images were loaded correctly
if normal_img is None or fracture_img is None:
    raise ValueError("Error: One or both images were not found. Check file paths.")

# Combine dataset
X = np.array([normal_img, fracture_img])
y = np.array([normal_label, fracture_label])

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

# 3. Train-Test Split (Only 2 images, so splitting is unnecessary)
X_train, X_test, y_train, y_test = X_features, X_features, y, y  # Using all data for both

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
