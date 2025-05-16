# src/modeling.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

def train_model(model_name, preprocessor, x_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'XGBoost': XGBRegressor()
    }
    model = models[model_name]
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(x_train, y_train)
    return pipeline