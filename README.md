
# 🧠 Hair Disease Detection using CNN + GUI

This project is focused on detecting different scalp/hair diseases such as Alopecia Areata, Head Lice, Psoriasis, and Folliculitis using a Convolutional Neural Network (CNN) along with image preprocessing (denoising + CLAHE). It also includes a Tkinter-based GUI for real-time image prediction.

## 📁 Dataset & Labeling
Dataset used: Images categorized into 4 classes:
Alopecia areata
Head Lice
Psoriasis
Folliculitis
Labels were originally YOLO-formatted.
Converted into folders for CNN using custom preprocessing and label-parsing logic.

## 🧪 Preprocessing Steps
- ✅ Image Denoising using cv2.fastNlMeansDenoisingColored
- ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
- ✅ Resized images to (224x224)
- ✅ Normalized pixel values to [0, 1]

## 🧠 Model Architecture
- Built using TensorFlow / Keras
- Includes:
   - Convolutional Layers
    - MaxPooling
    - Dropout
    - Fully Connected (Dense) layers
- Trained with ImageDataGenerator for data augmentation


## 📊 Training
- Trained for 25 epochs
- Tracked training & validation accuracy/loss
- Best epoch selected based on validation accuracy
- Model saved as cnn_model.h5

## 🖥 GUI (Tkinter)
- Select an image using file dialog
- Displays the selected image
- Outputs:
    - Predicted disease class
    - Confidence score
## 🚀 Features

- ✅ Detects 4 types of common hair/scalp diseases.
- ✅ Uses advanced preprocessing: Denoising + CLAHE.
- ✅ Integrated with a CNN model for classification.
- ✅ Clean and interactive Tkinter GUI for predictions.
- ✅ Model trained and saved in .h5 format for easy deployment.


# 🧠 Hair Disease Detection using CNN + GUI

This project is focused on detecting different scalp/hair diseases such as Alopecia Areata, Head Lice, Psoriasis, and Folliculitis using a Convolutional Neural Network (CNN) along with image preprocessing (denoising + CLAHE). It also includes a Tkinter-based GUI for real-time image prediction.

## 📁 Dataset & Labeling
Dataset used: Images categorized into 4 classes:
Alopecia areata
Head Lice
Psoriasis
Folliculitis
Labels were originally YOLO-formatted.
Converted into folders for CNN using custom preprocessing and label-parsing logic.

## 🧪 Preprocessing Steps
- ✅ Image Denoising using cv2.fastNlMeansDenoisingColored
- ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization) for local contrast enhancement
- ✅ Resized images to (224x224)
- ✅ Normalized pixel values to [0, 1]

## 🧠 Model Architecture
- Built using TensorFlow / Keras
- Includes:
   - Convolutional Layers
    - MaxPooling
    - Dropout
    - Fully Connected (Dense) layers
- Trained with ImageDataGenerator for data augmentation


## 📊 Training
- Trained for 25 epochs
- Tracked training & validation accuracy/loss
- Best epoch selected based on validation accuracy
- Model saved as cnn_model.h5

## 🖥 GUI (Tkinter)
- Select an image using file dialog
- Displays the selected image
- Outputs:
    - Predicted disease class
    - Confidence score
## Directory Usage

```bash
 Directory structure:
└── nav2924-digital_image_processing/
    ├── data.yaml
    ├── gui_hd.py
    ├── index.ipynb
    ├── custom/
    └── output/
```

## Installation
```bash
git clone https://github.com/nav2924/Digital_Image_Processing
```
```bash
pip install Requirements.txt
```

## Run the server 
```bash
python gui_predictor.py
```

## 🧰 Requirements
- Python 3.8+
- OpenCV
- NumPy
- TensorFlow
- Pillow (PIL)
- Tkinter (comes with Python)

## Authors

- [@SoumyadipRoy](https://github.com/SoumyadipRoy17)
- [@Naveel Vk](https://github.com/Naveel-VK)
- [@Zeba KP](https://github.com/zeba262)

