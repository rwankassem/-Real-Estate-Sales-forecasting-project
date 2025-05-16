# src/modeling.py
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from xgboost import XGBRegressor ,plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def xgb_train_and_evaluate (preprocessor,x_train,x_test,y_train,y_test):
    numeric_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = x_train.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
    ])

    # Apply log transform to y
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    pipeline.fit(x_train, y_train_log)

    # Predict on test data
    y_pred_log = pipeline.predict(x_test)  # Use x_test, not x_test_scaled_df

    # Inverse transform predictions
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse,r2,mae

def performance_comparison(data):
    results_df = pd.DataFrame(data)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16)

    # Plot Mean Squared Error
    axs[0].bar(results_df['model_name'], results_df['MSE'], color='skyblue')
    axs[0].set_title('Mean Squared Error (MSE)')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    axs[0].set_ylabel('MSE')

    # Plot R² Score
    axs[1].bar(results_df['model_name'], results_df['R2'], color='lightgreen')
    axs[1].set_title('R² Score')
    axs[1].set_ylabel('R²')

    # Plot Mean Absolute Error
    axs[2].bar(results_df['model_name'], results_df['MAE'], color='salmon')
    axs[2].set_title('Mean Absolute Error (MAE)')
    axs[2].set_ylabel('MAE')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def evaluate_classifiers(bins,x_train,x_test,y_train,y_test,numeric_cols,categorical_cols):
    n_bins = bins 

    # Create a KBinsDiscretizer instance
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    # Fit the discretizer on the training data and transform both train and test targets
    y_train_encoded = discretizer.fit_transform(y_train.values.reshape(-1, 1)).astype(int).ravel()
    y_test_encoded = discretizer.transform(y_test.values.reshape(-1, 1)).astype(int).ravel()

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Fit and evaluate
    for name, model in models.items():
        # Create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

        # Fit the pipeline
        pipeline.fit(x_train, y_train_encoded)

        # Predict
        y_pred = pipeline.predict(x_test)

        print(f'--- {name} ---')
        print(f'Accuracy: {accuracy_score(y_test_encoded, y_pred):.4f}')
        print('\nClassification Report:\n', classification_report(y_test_encoded, y_pred))
        print('Confusion Matrix:\n', confusion_matrix(y_test_encoded, y_pred))
        print('\n')

def plot_importance(x_train,x_train_scaled_df,y_train):
    model = XGBRegressor()
    model.fit(x_train_scaled_df, y_train)

    importance = model.feature_importances_
    for col, score in zip(x_train.columns, importance):
        print(f"{col}: {score:.4f}")


    plot_importance(model)
    plt.show()

def feature_selection(k,x_train,x_train_scaled,y_train):
   selector = SelectKBest(score_func=f_regression, k=k)
   X_new = selector.fit_transform(x_train_scaled, y_train)

   selected_features = x_train.columns[selector.get_support()]
   return selected_features.tolist() 

def RFE(x_train_scaled_df,y_train):
    model = XGBRegressor()
    rfe = RFE(estimator=model, n_features_to_select=10)
    rfe.fit(x_train_scaled_df, y_train)

    selected_features = x_train_scaled_df.columns[rfe.support_]
    print("Selected Features:", selected_features.tolist())
