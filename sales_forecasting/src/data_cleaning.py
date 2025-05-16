# src/data_cleaning.py
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=['List Price'])

    numeric_cols = ['Year Built', 'Bathrooms', 'Bedrooms', 'Square Footage']
    for col in numeric_cols:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    zip_mode = df['Zip'].mode()[0]
    df['Zip'].fillna(zip_mode, inplace=True)

    object_cols = ['State', 'Date Added', 'Status Change Date']
    for col in object_cols:
       df[col].fillna('mode', inplace=True)

    return df