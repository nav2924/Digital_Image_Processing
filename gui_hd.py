import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("model/cnn_model.h5")  # Replace with your model file
class_names = ["Alopecia areata", "Head_Lice", "Psoriasis", "Folliculitis"]  # Adjust your class names

IMG_SIZE = (224, 224) 

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # CLAHE
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize and normalize
    resized = cv2.resize(enhanced, IMG_SIZE)
    resized = resized / 255.0 
    return np.expand_dims(resized, axis=0)  

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))  
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk  

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        label_result.config(text=f"Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.2f}")

root = tk.Tk()
root.title("Hair Disease Detection")

btn_open = tk.Button(root, text="Select Image", command=open_image)
btn_open.pack()

label_img = tk.Label(root)
label_img.pack()

label_result = tk.Label(root, text="Prediction: ", font=("Arial", 14))
label_result.pack()

root.mainloop()
