
# src/evaluation.py
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred)
    }
