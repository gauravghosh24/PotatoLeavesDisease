# 🥔 Potato Leaf Disease Detection using Deep Learning

A Convolutional Neural Network (CNN) based image classification system that detects diseases in potato leaves using the PlantVillage dataset. The model is trained to classify leaf images into different disease categories and can assist in early disease identification for better crop management.

---

## 📌 Project Overview

Plant diseases significantly affect crop yield and quality. Early detection helps farmers take preventive measures and reduce losses.

This project uses Deep Learning and Computer Vision techniques to automatically identify potato leaf diseases from leaf images.

---

## 🎯 Objectives

- Classify potato leaf images into disease categories.
- Reduce manual inspection efforts.
- Demonstrate the use of CNNs for agricultural applications.
- Build a foundation for future deployment as a web or mobile application.

---

## 📂 Dataset

Dataset Source:

- PlantVillage Dataset (Kaggle)

The dataset contains labeled images of potato leaves belonging to the following classes:

1. Potato___Early_blight
2. Potato___Late_blight
3. Potato___healthy

Images are automatically loaded using TensorFlow's `image_dataset_from_directory()` API.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

---

## 📊 Data Preprocessing

### Image Loading

```python
tf.keras.preprocessing.image_dataset_from_directory()
```

### Image Size

```python
256 × 256 × 3
```

### Dataset Split

| Dataset | Percentage |
|----------|------------|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

### Performance Optimization

- Dataset Caching
- Prefetching
- Shuffling

```python
dataset.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
```

---

## 🔄 Data Augmentation

To reduce overfitting and improve generalization:

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation

```python
layers.RandomFlip("horizontal_and_vertical")
layers.RandomRotation(0.2)
```

---

## 🧠 Model Architecture

The model is built using a Sequential CNN architecture.

### Architecture

```text
Input Layer (256x256x3)

↓
Rescaling (1/255)

↓
Data Augmentation

↓
Conv2D (32 filters) + ReLU
MaxPooling2D

↓
Conv2D (64 filters) + ReLU
MaxPooling2D

↓
Conv2D (64 filters) + ReLU
MaxPooling2D

↓
Conv2D (64 filters) + ReLU
MaxPooling2D

↓
Conv2D (64 filters) + ReLU
MaxPooling2D

↓
Flatten

↓
Dense Layers

↓
Softmax Output Layer
```

---

## ⚙️ Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🚀 Training

```python
history = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds
)
```

---

## 📈 Performance Evaluation

The model performance is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

Visualization:

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

---

## 📷 Sample Predictions

The trained model can classify potato leaf images as:

| Disease | Description |
|----------|------------|
| Early Blight | Fungal disease causing dark spots |
| Late Blight | Serious disease causing rapid plant damage |
| Healthy | No disease detected |

---

## 📁 Project Structure

```text
Potato-Leaf-Disease-Detection/
│
├── potato_leave_disease.ipynb
├── README.md
├── models/
│   └── potato_disease_model.keras
│
├── dataset/
│
└── images/
```

---

## 🔮 Future Improvements

- Transfer Learning using MobileNetV2
- Grad-CAM Visualization
- Flask/FastAPI Deployment
- Real-time Disease Detection
- Mobile Application Integration
- Disease Treatment Recommendation System

---

## 💡 Applications

- Smart Agriculture
- Precision Farming
- Crop Health Monitoring
- Agricultural Research
- Farmer Assistance Systems

---

## 🤝 Acknowledgements

- PlantVillage Dataset
- TensorFlow
- Keras
- Kaggle Community

---

## 📜 License

This project is intended for educational and research purposes.

---

### ⭐ If you found this project useful, consider giving it a star on GitHub!
