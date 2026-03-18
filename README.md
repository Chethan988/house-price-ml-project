# 🏠 House Price Prediction using Machine Learning

## 📌 Project Overview
This project predicts house prices using machine learning models based on various features such as income, house age, rooms, and location.

---

## 📊 Dataset
- California Housing Dataset
- Features include:
  - Median Income
  - House Age
  - Average Rooms
  - Location (Latitude, Longitude)

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

---

## 🤖 Models Used
- Linear Regression
- Random Forest Regressor

---

## 📈 Model Performance

| Model              | R² Score | MSE |
|------------------|--------|------|
| Linear Regression | 0.57   | 0.55 |
| Random Forest     | 0.80   | 0.25 |

👉 Random Forest performed better due to its ability to capture non-linear relationships.

---

## 🔍 Key Insights
- Median Income is the most important feature
- Location (Latitude & Longitude) significantly impacts pricing
- Random Forest handles complex patterns better than Linear Regression

---

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn
python day13_house_price_prediction.py

## 📷 Project Output

### Model Comparison
- Linear Regression R²: 0.57  
- Random Forest R²: 0.80  

### Feature Importance (Random Forest)
Top features:
- Median Income
- Average Occupancy
- Location (Latitude & Longitude)

## 📁 Project Structure

house-price-ml-project/
│── day13_house_price_prediction.py
│── README.md
│── .gitignore

## 💡 Problem Statement
Predict housing prices based on socio-economic and geographical features using regression models.

## 🎯 Objective
Build and compare machine learning models to identify the best-performing algorithm for price prediction.
