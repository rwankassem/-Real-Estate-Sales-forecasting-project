import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Visualizing Real Estate Data")

DATA_PATH = 'data/cleaned_dataset.csv'  

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if 'City freq' in df.columns:
        df = df.drop(columns=['City freq'])
    return df

df = load_data(DATA_PATH)

st.success(f"Loaded dataset with {len(df)} records.")

# --- Visualizations and Statistics for entire dataset ---
st.subheader("Visualizations and Statistics for Entire Dataset")
st.markdown("---")

# 1- Correlation with List Price
st.title("Exploring Relations Between Numerical Features and Price")
st.subheader("Correlation with List Price")

numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()
correlation_list_price = correlation_matrix['List Price'].sort_values(ascending=False)

fig = px.bar(correlation_list_price, title='Correlation with List Price', labels={'value': 'Correlation', 'index': 'Feature'})
st.plotly_chart(fig)

# 2- Price Distribution by Categories
st.subheader("Price Distribution by Categories")

categorical_cols = ['House Size', 'House Type', 'State', 'City']
for col in categorical_cols:
    fig = px.box(df, x=col, y='List Price', title=f'Price Distribution by {col}')
    st.plotly_chart(fig)

# 3- Average List Price by Category
st.subheader("Average List Price by Category")
for col in categorical_cols:
    avg_price = df.groupby(col)['List Price'].mean().reset_index()
    avg_price = avg_price.sort_values('List Price', ascending=False)

    fig = px.bar(avg_price, x=col, y='List Price',
                 title=f'Average List Price by {col}',
                 labels={col: col, 'List Price': 'Average List Price'},
                 color='List Price', color_continuous_scale='Viridis')

    fig.update_layout(
        xaxis_title=col,
        yaxis_title='Average List Price',
        template='plotly_white',
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig)
    st.markdown("---")

# 4- Observations 
st.subheader("Observations")
st.markdown("""
- **State**: North VA and DC have the highest prices.
- **House Size**: Larger house sizes show a wider price range and more outliers.
- **House Type**: Single-family home has a great impact on the price.
- **City**: North Bethesda is the highest priced city.
""")
st.markdown("---")

# 5- Numeric categorical relations
st.title("Exploring Relations Between Numerical Features and Price")
numeric_cols = ['List Price', 'Bathrooms', 'Bedrooms', 'House Age', 'Year Built']
num_categorical_cols = ['Bathrooms', 'Bedrooms', 'House Age', 'Year Built']

for col in num_categorical_cols:
    avg_price = df.groupby(col)['List Price'].mean().reset_index()
    avg_price = avg_price.sort_values('List Price', ascending=False)

    fig = px.bar(avg_price, x=col, y='List Price',
                 title=f'Average List Price by {col}',
                 labels={col: col, 'List Price': 'Average List Price'},
                 color='List Price', color_continuous_scale='Viridis')

    fig.update_layout(
        xaxis_title=col,
        yaxis_title='Average List Price',
        template='plotly_white',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)
    st.markdown("---")

st.subheader("Observations")
st.markdown("""
- **Bathrooms – 5**: Houses with 5 bathrooms have an average list price of $1,000,247.
- **Bedrooms – 6**: Properties with 6 bedrooms have the highest average price of $846,249.
- **House Age – 92 years**: Homes built around 1933 also have a high average price.
- **Year Built – 1933**: Houses built in 1933 have the highest average list price of $1,238,140.
""")
st.markdown("---")

# 6- Basic statistics for entire dataset
st.subheader("Basic Statistics for Entire Dataset")
st.write(df.describe())
st.markdown("---")

# 7- Filtered Data Analysis
st.subheader("Visualizations and Statistics for Filtered Data")

states = df['State'].unique()
selected_state = st.selectbox("Select State", states)
state_filtered_df = df[df['State'] == selected_state]

cities = state_filtered_df['City'].unique()
selected_city = st.selectbox("Select City", cities)
city_filtered_df = state_filtered_df[state_filtered_df['City'] == selected_city]

house_types = city_filtered_df['House Type'].unique()
selected_type = st.selectbox("Select House Type", house_types)
type_filtered_df = city_filtered_df[city_filtered_df['House Type'] == selected_type]

house_sizes = type_filtered_df['House Size'].unique()
selected_size = st.selectbox("Select House Size", house_sizes)
filtered_df = type_filtered_df[type_filtered_df['House Size'] == selected_size]

st.subheader("Filtered Data")
st.dataframe(filtered_df)
st.info(f"Number of available properties matching your filters: {len(filtered_df)}")

fig = px.histogram(filtered_df, x='List Price', title=f"Price Distribution in {selected_city} - {selected_type} - {selected_size}")
st.plotly_chart(fig)

fig2 = px.box(filtered_df, x='House Size', y='List Price', title="Price by House Size")
st.plotly_chart(fig2)

fig3 = px.box(filtered_df, x='House Type', y='List Price', title="Price by House Type")
st.plotly_chart(fig3)

fig4 = px.box(filtered_df, x='City', y='List Price', title="Price by City")
st.plotly_chart(fig4)

# Correlation matrix for filtered data
correlation_matrix = filtered_df.select_dtypes(include=['int64', 'float64']).corr()
fig5 = px.imshow(correlation_matrix, title="Correlation Matrix", color_continuous_scale='Viridis')
st.plotly_chart(fig5)
st.markdown("---")

st.subheader("Basic Statistics for Filtered Data")
st.write(filtered_df.describe())
