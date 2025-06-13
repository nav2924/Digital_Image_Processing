# 🧠 Hair Disease Detection System using Deep Learning

An intelligent and efficient system designed to detect and localize scalp-related diseases using deep learning and computer vision techniques. This project combines **CNN-based classification**, **image preprocessing (CLAHE)**, and **YOLOv8-based object detection** into a single, user-friendly platform.

🌐 **Live Demo:** [Streamlit App](https://hair-health.onrender.com/)  
📦 **GitHub Repository:** https://github.com/nav2924/hair-disease-detection

---

## 🚀 Features

- 🔍 **Advanced Preprocessing Pipeline:** Denoising and CLAHE (Contrast Limited Adaptive Histogram Equalization) improve image clarity and contrast.
- 🧠 **Deep Learning Classification:** CNN trained on enhanced and balanced datasets for accurate disease prediction.
- 🎯 **YOLOv8 Detection:** Localizes affected scalp regions for precise diagnosis assistance.
- 🖼️ **Interactive UI:** Streamlit-powered interface for easy image upload and instant feedback.
- 📊 **Optimized Accuracy:** Achieved via data augmentation, hyperparameter tuning, and proper validation techniques.

---

## 🧪 Tech Stack

| Category           | Tools / Libraries                              |
|--------------------|-------------------------------------------------|
| Language           | Python                                          |
| Deep Learning      | TensorFlow, Keras                               |
| Image Processing   | OpenCV, CLAHE                                   |
| Object Detection   | YOLOv8 (Ultralytics)                            |
| User Interface     | Streamlit                                       |
| Utilities          | NumPy, Matplotlib, scikit-learn                 |

---

## 🧬 System Architecture

```text
Raw Image → Preprocessing (CLAHE + Denoising) → CNN Classifier → YOLOv8 Detector → Output with Mask & Prediction
```

---

## 💻 Getting Started

Clone the repository and run the application locally:

```bash
git clone https://github.com/nav2924/hair-disease-detection.git
cd hair-disease-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## 📸 Sample Output
![image](https://github.com/user-attachments/assets/52cf7499-3dca-47a5-a8a5-d9f5a5fde911)
![image](https://github.com/user-attachments/assets/2c52fe9d-7a99-49b0-b151-315ff0567b4a)


---

## 👨‍💻 Team

- [Naveen V K](https://www.linkedin.com/in/naveen-v-k)
- [Zeba K P](https://www.linkedin.com/in/zeba-k-p)
- [Naveel Vk](https://www.linkedin.com/in/naveel-vk)

---

## 🔭 Future Enhancements

- 📱 **Mobile Deployment:** Integration with Flutter or React Native for Android/iOS support.
- 🧬 **Dataset Expansion:** Inclusion of more diverse hair and scalp disease categories.
- 🧑‍⚕️ **Clinical Collaboration:** Partnering with dermatologists for expert validation and feedback.
- 📈 **Explainability:** Add Grad-CAM visualizations for model interpretability.

---

## 🙏 Acknowledgements

We thank our team members for their dedicated contributions, collaborative energy, and relentless debugging efforts that made this project a success.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
