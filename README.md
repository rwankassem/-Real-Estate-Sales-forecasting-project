# -Real-Estate-Sales-forecasting-project
Project Name: 
    Sales forecasting and price prediction

    
Team Members:
-    Rwan Taha (Team Leader)
-    Asmaa Gamal
-    Sara Ashraf
-    Mohamed Abdelhamid


This project predicts house prices using machine learning models like **XGBoost**, **Random Forest**, and **Linear Regression**, based on real estate data.

It also includes a fully interactive **Streamlit dashboard** for data exploration and visualization.

---

##  Features
- Data Cleaning & Preprocessing
- Feature Engineering (House Age, House Size, City Frequency)
- Outlier Detection & Handling
- Correlation Analysis & EDA
- ML Modeling (XGBoost, Random Forest, Linear Regression)
- Streamlit App for Visualization & Filtering

---

##  Project Structure
```
sales_forecasting/
│
├── app.py                  # Streamlit dashboard
├── src/                    # Scripts
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── modeling.py
│   └── evaluation.py
├── data/
│   ├── DMV_Homes.csv           # Raw dataset
│   └── cleaned_dataset.csv     # Cleaned dataset after preprocessing
├── requirements.txt
└── README.md
```

---

##  How to Run the Project

1. **Clone the repo:**
```bash
git clone https://github.com/rwankassem/-Real-Estate-Sales-forecasting-project.git
```

2. **Install the requirements:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

---

##  Model Performance
| Model              | R² Score | MAE        |
|-------------------|----------|------------|
| Linear Regression | 0.07     | 399,720    |
| Random Forest     | 0.60     | 204,674    |
| **XGBoost**       | **0.79** | **225,574** |

---

##  Future Improvements
- Deploy app on Streamlit Cloud
- Add LightGBM model
- Integrate API for real-time data

---
=======



>>>>>>> 83a530720ad4bece0dcc491ea5f401a48d09a7b5
