#  Real-Estate-Sales-forecasting-project

##  Project Name: 
**Sales Forecasting and Price Prediction**

## Team Members:
- Rwan Taha (Team Leader)
- Asmaa Gamal
- Sara Ashraf
- Mohamed Abdelhamid

---

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
├── data/                   # Datasets
│   ├── DMV_Homes.csv       # Raw dataset
│   └── cleaned_dataset.csv # Cleaned dataset
├── notebooks/              # Notebooks and scripts
│   └── sales_forecasting.py
├── deployment/             # Deployment files
│   └── Deployment Django.zip
├── presentation/           # Presentation files
│   └── Presentation.pptx
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
streamlit run app/streamlit_app.py
```

---

## Model Performance
| Model              | R² Score | MAE        |
|-------------------|----------|------------|
| Linear Regression | 0.07     | 399,720    |
| Random Forest     | 0.60     | 204,674    |
| **XGBoost**       | **0.803** | **225,574** |

---

##  Future Improvements
- Deploy app on Streamlit Cloud
- Integrate real-time property listing API

---

##  Visual Examples 

### Categorical Feature Association (Cramér's V)
*Shows correlation between categorical features.*
---

###  PCA Projection
*Dimensionality reduction to visualize house clusters based on price.*
---

###  ANOVA Feature Importance
*Which categorical features have the most impact on price.*
---

###  Model Comparison
*Evaluation of models using R², MAE, MSE.*
---

### 🌐Streamlit App Interface
*Upload data and get real-time price predictions.*
---

###  Deployment Architecture
*How the ML model was integrated into Django backend and served through Streamlit.*
---

##  Technologies Used

- **Python 3.10** – Core programming language
- **Pandas / NumPy** – Data manipulation
- **Matplotlib / Seaborn / Plotly** – Visualization
- **Scikit-learn** – Machine learning algorithms & preprocessing
- **XGBoost / Random Forest / Linear Regression** – Modeling
- **SHAP** – Feature importance & interpretability
- **Streamlit** – Interactive web application
- **Git + GitHub** – Version control & collaboration
