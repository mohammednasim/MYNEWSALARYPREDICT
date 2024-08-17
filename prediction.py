import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

@st.cache_data

# Load and preprocess the data
def load_and_preprocess_data():
    data = pd.read_csv('survey_results_public.csv')

    # Select relevant columns
    df = data[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedComp']]
    df = df.rename({'ConvertedComp': 'Salary'}, axis=1)

    # Drop missing values
    df = df[df['Salary'].notnull()]
    df = df.dropna()

    # Filter for full-time employment
    df = df[df['Employment'] == 'Employed full-time'].drop('Employment', axis=1)

    # Map country categories
    country_map = digest_categories(df['Country'].value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)

    # Filter salaries within a specific range
    df = df[(df['Salary'] <= 250000) & (df['Salary'] > 10000)]
    df = df[df['Country'] != 'Other']

    # Map experience levels
    df['YearsCodePro'] = df['YearsCodePro'].apply(set_experience)

    # Reduce education levels
    df['EdLevel'] = df['EdLevel'].apply(set_education)

    return df


# Helper functions
def digest_categories(categories, cutoff):
    categorical_map = {cat: (cat if count >= cutoff else 'Other') for cat, count in categories.items()}
    return categorical_map


def set_experience(i):
    if i == 'More than 50 years':
        return 50
    if i == 'Less than 1 year':
        return 0.5
    return float(i)


def set_education(i):
    if 'Master’s degree' in i:
        return 'Master’s degree'
    if 'Bachelor’s degree' in i:
        return 'Bachelor’s degree'
    if 'Professional degree' in i or 'Other doctoral degree' in i:
        return 'Post grad'
    return 'Less than a Bachelors'


# Train model
def train_model(df):
    X = df[['Country', 'EdLevel', 'YearsCodePro']]
    X = pd.get_dummies(X)
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train.columns


# Make predictions
def predict_salary(model, columns, country, ed_level, years_exp):
    x = pd.DataFrame([[country, ed_level, years_exp]], columns=['Country', 'EdLevel', 'YearsCodePro'])
    x = pd.get_dummies(x)

    # Ensure all columns present during training are in the prediction input
    x = x.reindex(columns=columns, fill_value=0)

    salary_pred = model.predict(x)[0]
    return salary_pred


# Streamlit app
def main():
    st.title("SDE Salary Prediction")

    df = load_and_preprocess_data()
    model, columns = train_model(df)

    st.write("Select your inputs:")

    country = st.selectbox('Country', df['Country'].unique())
    ed_level = st.selectbox('Education Level', df['EdLevel'].unique())
    years_exp = st.slider('Years of Professional Coding Experience', 0, 50, 5)

    if st.button('Predict Salary'):
        salary_pred = predict_salary(model, columns, country, ed_level, years_exp)
        st.write(f"Predicted Salary: ${salary_pred:,.2f}")


if __name__ == '__main__':
    main()
