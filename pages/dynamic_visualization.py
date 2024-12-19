import streamlit as st
import pandas as pd
import plotly.express as px

# Sample DataFrame (you can load your own data here)
data = {
    'Date': pd.date_range(start='2020-01-01', periods=10, freq='D'),
    'Sales': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
    'Profit': [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West', 'North', 'South'],
    'Size': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]  # Size for Bubble chart
}

df = pd.DataFrame(data)

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Function to categorize column types into Numeric, Categorical, Date
def categorize_column_types(df):
    column_types = {}
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            column_types[column] = 'Numeric'
        elif pd.api.types.is_categorical_dtype(dtype) or dtype == 'object':
            column_types[column] = 'Categorical'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[column] = 'Date'
        else:
            column_types[column] = 'Other'
    return column_types

# Streamlit UI Components
st.title('Dynamic Column & Chart Selection')

# Categorize columns based on data types
column_types = categorize_column_types(df)

# Display column types to the user
st.write("### Column Types (Numeric, Categorical, Date)")
st.write(column_types)

# Let the user select the X-axis and Y-axis columns
x_axis_column = st.selectbox('Select X-axis Column', df.columns)
y_axis_column = st.selectbox('Select Y-axis Column', df.columns)

# Select the type of chart
chart_type = st.selectbox('Select Chart Type', ['Bar', 'Line', 'Scatter', 'Pie', 'Horizontal Bar', 'Histogram', 'Bubble'])

# Define a custom color scale for charts
color_scale = ['#00A3E0', '#008C87', '#FF6F61', '#D76D77', '#9B59B6', '#F39C12']

# Check for missing data in the selected columns
def check_missing_data(columns):
    return df[columns].isnull().any().any()

# Ensure the selected columns are appropriate for the chart type
def check_column_types(chart_type, x_column, y_column):
    if chart_type in ['Bar', 'Line', 'Scatter']:
        # X-axis should be categorical for Bar/Line, and Y-axis should be numerical
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return False, "Y-axis must be numerical for this chart type."
        if chart_type == 'Bar' and not (pd.api.types.is_categorical_dtype(df[x_column]) or df[x_column].dtype == 'object'):
            return False, "X-axis should be categorical for Bar chart."
    elif chart_type == 'Pie':
        # Pie chart needs categorical X and numerical Y
        if not pd.api.types.is_categorical_dtype(df[x_column]) and df[x_column].dtype != 'object':
            return False, "X-axis should be categorical for Pie chart."
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            return False, "Y-axis must be numerical for Pie chart."
    elif chart_type == 'Histogram':
        # Histogram requires a numerical column for X
        if not pd.api.types.is_numeric_dtype(df[x_column]):
            return False, "X-axis must be numerical for Histogram chart."
    elif chart_type == 'Bubble':
        # Bubble chart requires a numerical X, Y, and a size column
        if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
            return False, "X and Y axes must be numerical for Bubble chart."
        if 'Size' not in df.columns or not pd.api.types.is_numeric_dtype(df['Size']):
            return False, "Bubble chart requires a 'Size' column with numerical data."
    return True, ""

# Debugging: Display selected parameters
st.write("Selected X-axis column:", x_axis_column)
st.write("Selected Y-axis column:", y_axis_column)
st.write("Selected chart type:", chart_type)

# Perform checks for missing data and column type validation
if check_missing_data([x_axis_column, y_axis_column]):
    st.write("Warning: One or more of the selected columns contain missing data. Please clean the data.")
else:
    valid, error_message = check_column_types(chart_type, x_axis_column, y_axis_column)
    if not valid:
        st.write(f"Error: {error_message}")
    else:
        # Proceed with chart generation
        try:
            if chart_type == 'Bar':
                fig = px.bar(df, x=x_axis_column, y=y_axis_column, title=f'{chart_type} Chart', color=df[y_axis_column], 
                             color_continuous_scale=color_scale)
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title=y_axis_column,
                    legend_title="Legend"
                )
            elif chart_type == 'Line':
                fig = px.line(df, x=x_axis_column, y=y_axis_column, title=f'{chart_type} Chart', color='Region')  # Using 'Region' for color
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title=y_axis_column,
                    legend_title="Region"
                )
            elif chart_type == 'Scatter':
                fig = px.scatter(df, x=x_axis_column, y=y_axis_column, title=f'{chart_type} Chart', color=df[y_axis_column],
                                 color_continuous_scale=color_scale)
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title=y_axis_column,
                    legend_title="Legend"
                )
            elif chart_type == 'Pie':
                fig = px.pie(df, names=x_axis_column, values=y_axis_column, title=f'{chart_type} Chart', color=df[y_axis_column],
                             color_discrete_sequence=color_scale)
                fig.update_layout(
                    legend_title="Legend"
                )
            elif chart_type == 'Horizontal Bar':
                fig = px.bar(df, x=y_axis_column, y=x_axis_column, orientation='h', title=f'{chart_type} Chart',
                             color=df[y_axis_column], color_continuous_scale=color_scale)
                fig.update_layout(
                    xaxis_title=y_axis_column,
                    yaxis_title=x_axis_column,
                    legend_title="Legend"
                )
            elif chart_type == 'Histogram':
                fig = px.histogram(df, x=x_axis_column, title=f'{chart_type} Chart', color=df[y_axis_column],
                                    color_discrete_sequence=color_scale)
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title="Count",
                    legend_title="Legend"
                )
            elif chart_type == 'Bubble':
                fig = px.scatter(df, x=x_axis_column, y=y_axis_column, size='Size', title=f'{chart_type} Chart',
                                 color=df[y_axis_column], color_continuous_scale=color_scale)
                fig.update_layout(
                    xaxis_title=x_axis_column,
                    yaxis_title=y_axis_column,
                    legend_title="Legend"
                )

            # Display the figure
            st.plotly_chart(fig)

        except Exception as e:
            st.write(f"Error generating the chart: {str(e)}")
