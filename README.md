# Brain Tumor Detection using Deep Learning

## 📌 Overview
This project is a **Brain Tumor Detection** model built using **Convolutional Neural Networks (CNNs)** in TensorFlow/Keras. It classifies MRI images into **Tumor** and **No Tumor** categories.

## 🚀 Model Information
- **Model Type:** CNN (Convolutional Neural Network)
- **Input Shape:** (128, 128, 3)
- **Activation Function:** ReLU & Softmax
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adamax
- **Dataset:** Brain MRI Images (Yes/No Tumor)

## 🔗 Model File
You can download the trained model from the following link:
📥 **[Download Model](https://drive.google.com/file/d/1I1q4y0O4af2muT5n7iI3s7ehhl7NMCPz/view?usp=drive_link)**

## 📂 Dataset Structure
```
brain_tumor_dataset/
│── yes/   # Images with tumors
│── no/    # Images without tumors
```

## ⚡ Installation
To set up the project, install the dependencies:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## 🏗 Model Architecture
```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding='Same'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2,2), activation='relu', padding='Same'))
model.add(Conv2D(64, kernel_size=(2,2), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adamax')
```

## 🏋️‍♂️ Training the Model
```python
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=25, batch_size=32)
```

## 📊 Performance Metrics
![download](https://github.com/user-attachments/assets/e3223f36-8ca2-4561-b821-28a85b6bdf9c)

## 💾 Saving & Loading the Model
```python
# Save the model in TensorFlow format
model.save("brain_tumor_model_tf", save_format="tf")

# Load the model
from tensorflow.keras.models import load_model
model = load_model("brain_tumor_model_tf")
```

## 🖼 Sample Prediction
```python
import cv2
img = cv2.imread("sample_image.jpg")
img = cv2.resize(img, (128, 128))
img = img.reshape(1, 128, 128, 3)
prediction = model.predict(img)
print("Tumor Detected" if np.argmax(prediction) == 1 else "No Tumor")
```

### 📢 If you find this project useful, don't forget to ⭐ star the repository!

