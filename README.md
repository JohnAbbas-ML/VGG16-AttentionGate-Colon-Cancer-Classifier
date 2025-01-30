# VGG16-AttentionGate based Colon Cancer Classifier

## Introduction
This project focuses on classifying colon cancer histopathological images using deep learning techniques. The primary goal is to develop a robust Convolutional Neural Network (CNN) model based on VGG16 with an Attention Gate mechanism to improve classification accuracy. The dataset used consists of colon histopathological images categorized into two classes: 
- **Colon Adenocarcinoma (colon_aca)**
- **Colon Benign Tissue (colon_n)**

The project is implemented using TensorFlow and Keras, and it aims to assist in the early detection and classification of colon cancer, which is crucial for timely treatment and improved survival rates.

## Proposed Methodology
### 1. **Data Loading and Preprocessing**
The dataset is loaded from the directory structure, and file paths with corresponding labels are stored in a Pandas DataFrame. The labels are then standardized to maintain consistency.

### 2. **Exploratory Data Analysis (EDA)**
- The distribution of classes is visualized using a pie chart to check for class imbalance.
- Sample images from each class are displayed to understand data characteristics.

### 3. **Data Splitting**
The dataset is divided into three subsets:
- **Training set (70%)**
- **Validation set (15%)**
- **Testing set (15%)**

### 4. **Data Augmentation**
To enhance generalization and prevent overfitting, data augmentation techniques are applied, including:
- Rescaling
- Brightness adjustment
- Rotation
- Zooming
- Horizontal flipping

### 5. **Building the CNN Model**
The model is based on VGG16 (pretrained on ImageNet) with:
- Attention Gate mechanism to enhance important features
- Regularization to prevent overfitting
- Fully connected layers with dropout for robust learning
- Sigmoid activation for binary classification

### 6. **Training and Optimization**
The model is compiled using:
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam with a learning rate of 0.001
- **Metrics:** Accuracy, Precision, Recall, and AUC

Early Stopping and Learning Rate Reduction are used to optimize training and prevent overfitting.

### 7. **Performance Evaluation**
The model is evaluated on training, validation, and testing sets using:
- Accuracy, Precision, Recall, and AUC metrics
- Training history visualization (loss, accuracy, precision, recall, and AUC curves)
- Confusion matrix to analyze classification performance

### 8. **Saving the Model**
The trained model is saved as `model.save('ColonCancer_VGGNet.h5')` for future inference and deployment.

## Testing and Results
### **Model Performance**
The model's performance is evaluated based on key metrics:
- **Training Accuracy:** Achieved high accuracy during training
- **Validation Accuracy:** Demonstrated good generalization ability
- **Testing Accuracy:** Confirmed the model's reliability on unseen data

### **Confusion Matrix Analysis**
A confusion matrix is used to assess the classification performance:
- True Positives (Correctly classified adenocarcinoma cases)
- True Negatives (Correctly classified benign cases)
- False Positives (Benign misclassified as adenocarcinoma)
- False Negatives (Adenocarcinoma misclassified as benign)

The results indicate a high precision and recall, showing that the model is well-suited for colon cancer classification.

### **Model Inference**
To use the trained model for predictions:
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('ColonCancer_VGGNet.h5')

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Colon Adenocarcinoma" if prediction[0][0] > 0.5 else "Colon Benign Tissue"

print(predict_image('path_to_test_image.jpg'))
```

## Conclusion
This deep learning-based approach successfully classifies colon cancer histopathological images, leveraging transfer learning with VGG16 and an Attention Gate mechanism. The model demonstrates high accuracy, precision, recall, and AUC, making it a valuable tool for assisting medical professionals in cancer diagnosis. Future improvements could involve experimenting with other architectures like ResNet or EfficientNet and further enhancing interpretability through explainable AI techniques.

