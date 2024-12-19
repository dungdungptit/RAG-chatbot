import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import matplotlib.pyplot as plt

# Synthetic generation of data examples for training the model
def generate_house_data(n_samples=100):
    np.random.seed(42)
    size = np.random.normal(1500, 500, n_samples)
    price = size * 100 + np.random.normal(0, 10000, n_samples)
    return pd.DataFrame({'size_sqft': size, 'price': price})

# Function for instantiating and training linear regression model
def train_model():
    df = generate_house_data()
    
    # Train-test data splitting
    X = df[['size_sqft']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Streamlit User Interface for Deployed Model
def main():
    st.title('🏠 Simple House Pricing Predictor')
    st.write('Introduce the house size to predict its sale price')
    
    # Train model
    model = train_model()
    
    # User input
    size = st.number_input('House size (square feet)', 
                          min_value=500, 
                          max_value=5000, 
                          value=1500)
    
    if st.button('Predict price'):
        # Perform prediction
        prediction = model.predict([[size]])
        
        # Show result
        st.success(f'Estimated price: ${prediction[0]:,.2f}')
        
        # Visualization
        df = generate_house_data()
        fig = px.scatter(df, x='size_sqft', y='price', 
                        title='Size vs Price Relationship')
        fig.add_scatter(x=[size], y=[prediction[0]], 
                       mode='markers', 
                       marker=dict(size=15, color='red'),
                       name='Prediction')
        st.plotly_chart(fig)
    
    df = generate_house_data()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols + date_cols]

    # Logic to choose the chart
    if 'Dates' in date_cols and len(numeric_cols) > 0:
        st.line_chart(df.set_index('Dates')[numeric_cols])
    elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
        st.bar_chart(df.set_index(categorical_cols[0])[numeric_cols])
    elif len(numeric_cols) > 1:
        st.line_chart(df[numeric_cols])
    else:
        st.write("Data is not suitable for a specific chart type.")

    # Dynamic column selection
    columns = df.columns.tolist()
    selected_column = st.selectbox("Select a column to visualize", columns)

    # Visualize the selected column
    st.subheader(f'Visualization of {selected_column}')
    # Plot a chart based on the selected column
    fig, ax = plt.subplots()
    
    # Check if the selected column is numeric for plotting
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        ax.plot(df[selected_column], marker='o', linestyle='-', color='b')
        ax.set_title(f'{selected_column} over Time')
        ax.set_xlabel('Index')
        ax.set_ylabel(selected_column)
    else:
        # For non-numeric columns, plot a bar chart of frequency counts
        value_counts = df[selected_column].value_counts()
        ax.bar(value_counts.index, value_counts.values, color='c')
        ax.set_title(f'Distribution of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Frequency')

    # Display the plot
    st.pyplot(fig)
    
if __name__ == '__main__':
    main()
    # https://machinelearningmastery.com/how-to-quickly-deploy-machine-learning-models-streamlit/
    # streamlit run ML_deploy.py