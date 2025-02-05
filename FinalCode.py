import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Function to load images from a folder with augmentation
def load_images_from_folder(folder, label):
    images, labels = [], []
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' not found.")
        return images, labels
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
            
            # Data augmentation
            images.append(cv2.flip(img, 1))  # Horizontal flip
            labels.append(label)
            
            images.append(cv2.GaussianBlur(img, (5,5), 0))  # Gaussian Blur
            labels.append(label)
    return images, labels

# Load training data
train_normal, train_normal_labels = load_images_from_folder("cervical fracture/train/normal", label=0)
train_fracture, train_fracture_labels = load_images_from_folder("cervical fracture/train/fracture", label=1)

# Load validation data
val_normal, val_normal_labels = load_images_from_folder("cervical fracture/val/normal", label=0)
val_fracture, val_fracture_labels = load_images_from_folder("cervical fracture/val/fracture", label=1)

# Combine train dataset
X_train = np.array(train_normal + train_fracture)
y_train = np.array(train_normal_labels + train_fracture_labels)

# Combine validation dataset
X_test = np.array(val_normal + val_fracture)
y_test = np.array(val_normal_labels + val_fracture_labels)

# Check if images were loaded correctly
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Error: No images found in one or both datasets. Check file paths.")

# Feature extraction using Sobel and HOG
def extract_features(img):
    # Sobel edge detection
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    
    sobel_features = [np.mean(sobel_combined), np.var(sobel_combined), np.sum(sobel_combined > 50)]
    
    # HOG feature extraction
    hog_features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    
    return np.hstack((sobel_features, hog_features))

# Extract features
X_train_features = np.array([extract_features(img) for img in X_train])
X_test_features = np.array([extract_features(img) for img in X_test])

# Standardize the features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_features, y_train)

# Best model training
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_features, y_train)

# Make predictions
y_pred = best_rf.predict(X_test_features)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualize Sobel Edge Detection on a sample image
sample_img = X_train[0]
sobelx = cv2.Sobel(sample_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(sample_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobelx, sobely)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1), plt.imshow(sample_img, cmap='gray'), plt.title("Original C5 Image")
plt.subplot(1, 3, 2), plt.imshow(sobelx, cmap='gray'), plt.title("Sobel X")
plt.subplot(1, 3, 3), plt.imshow(sobel_combined, cmap='gray'), plt.title("Sobel Edge Detection")
plt.show()
