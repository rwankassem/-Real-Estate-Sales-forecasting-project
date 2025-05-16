
# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

def get_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

def label_encoder(df):
    df_encoded = df.copy()
    le_house = LabelEncoder()
    df_encoded['House Type'] = le_house.fit_transform(df_encoded['House Type'])

    le_state=LabelEncoder()
    df_encoded['State'] = le_state.fit_transform(df_encoded['State'])

    return df_encoded    

def freq_encoder(df):
    df['City_freq'] = df['City'].value_counts()
    city_freq = df['City'].value_counts().to_dict()
    df['City_freq'] = df['City'].apply(lambda x: city_freq.get(x, 0) if pd.notna(x) else 0)

    return(df)

def split_data(df):
    x=df.drop(['List Price','City'],axis=1)
    y=df['List Price']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    numeric_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = x_train.select_dtypes(include='object').columns

    return x_train, x_test, y_train, y_test, numeric_cols, categorical_cols

def float_to_int(df,col_names):
    for col in col_names:
        df[col] = df[col].astype(int)
    return df

def apply_dbscan(df):
    data = df[['Square Footage', 'List Price']].values  

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Applying DBSCAN
    dbscan = DBSCAN(eps=.5, min_samples=5)  
    y_db = dbscan.fit_predict(data_scaled)

    plt.figure(figsize=(8, 6))

    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=y_db, cmap='viridis', marker='o', edgecolor='k')

    # Plotting the noise points (label == -1)
    plt.scatter(data_scaled[y_db == -1, 0], data_scaled[y_db == -1, 1], color='red', marker='x', label='Noise')

    plt.title('DBSCAN Clustering on Square Footage and List Price')
    plt.xlabel('Standardized Square Footage')
    plt.ylabel('Standardized List Price')
    plt.legend()
    plt.show()

    return y_db

def apply_clib (df,numeric_cols):
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)

         # Apply clipping
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)

def explore_outliers(df,cols):
    fig, axes = plt.subplots(1, len(cols), figsize=(12,6))

    for i,col in enumerate(cols) :
        sns.boxplot(data=df[col],ax=axes[i],color='skyblue')
        axes[i].set_title(f'{col}')

    plt.suptitle(f'Box plots for {cols}')
    plt.tight_layout()
    plt.show()

def apply_SelectKBest(x_train_transformed,x_test_transformed,y_train,k):

    selector = SelectKBest(score_func=f_regression, k=k)
    x_train_selected = selector.fit_transform(x_train_transformed, y_train)
    x_test_selected = selector.transform(x_test_transformed)

    return x_train_selected,x_test_selected