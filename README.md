# 🩺 Physiological Stress Detection API

[![Live Demo](https://img.shields.io/badge/Demo-Live%20Now-green?style=for-the-badge&logo=render)](https://nurse-stress-detection-cosmic.onrender.com)

A Deep Learning-powered web application that provides real-time stress assessment for healthcare professionals. Built using a Keras Neural Network and deployed as a Flask API.

---

## 🚀 Live Access
**Click the badge above or the link below to use the application:**
👉 **(https://nurse-stress-detection-cosmic.onrender.com)**

*(Note: If using a free-tier hosting service like Render or Railway, the initial load may take 30-60 seconds as the server wakes up.)*

---

## 🧠 How it Works
This system takes raw physiological data and processes it through a machine learning pipeline to determine mental and physical states.

---

## DATASET
👉 **(https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors)**

### **The Assessment Process:**
1. **Data Entry:** User provides EDA (Skin Conductance), Heart Rate, and Temperature.
2. **Preprocessing:** Data is normalized using a pre-fitted `StandardScaler`.
3. **Deep Learning Inference:** A Neural Network classifies the data into 3 states:
   - ✅ **Baseline:** Relaxed/Normal state.
   - ⚠️ **Stress Detected:** High-arousal/Negative stress state.
   - 🏃 **Physical Activity:** Amusement or physical exertion.
4. **Actionable Insights:** Based on the result, the system provides immediate health interventions and breathing exercises.

---

## 🛠️ Technical Stack
* **Language:** Python 3.12
* **Model:** Deep Neural Network (Sequential API)
* **Libraries:** TensorFlow, Keras, Scikit-Learn, Pandas, Numpy
* **Backend:** Flask (Web Server Gateway)
* **UI/UX:** HTML5, CSS3 (Modern Responsive Design)

---

## 📂 Project Assets
* `app.py`: The core engine handling data routing and model prediction.
* `stress_detection_model.keras`: The compiled weights of the trained AI.
* `stress_scaler.pkl`: The mathematical distribution used for data scaling.
* `templates/`: The frontend user interface.

---

## 👨‍💻 Developer
**Ankit Bajpayee** *B.Tech in Artificial Intelligence & Machine Learning* 

---
