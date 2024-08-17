import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.write("""
## Exploratory Data Analysis ans Visualization

""")

def digest_categories(categories,cutoff):
    categoricalMap={}
    for i in range(len(categories)):
        if categories.values[i]>=cutoff:
            categoricalMap[categories.index[i]] = categories.index[i]
        else:
            categoricalMap[categories.index[i]] ='Other'
    return categoricalMap

def setExperience(i):
    if i == 'More than 50 years':
        return 50
    if i == 'Less than 1 year':
        return 0.5
    return float(i)

def setEducation(i):
    if 'Master’s degree' in i:
        return 'Master’s degree'
    if 'Bachelor’s degree' in i:
        return 'Bachelor’s degree'
    if 'Professional degree' in i or 'Other doctoral degree' in i:
        return 'Post grad'
    return 'Less than a Bachelors'

@st.cache_data

def load_data():
    data = pd.read_csv('survey_results_public.csv')
    df = data[['Country', 'EdLevel', 'YearsCodePro', 'Employment', 'ConvertedComp']]
    df = df.rename({'ConvertedComp': 'Salary'}, axis=1)
    df = df[df['Salary'].notnull()]
    df = df.dropna()
    shaped = df[df['Employment'] == 'Employed full-time']
    df = shaped.drop('Employment', axis=1)
    countryMap = digest_categories(df.Country.value_counts(), 400)
    df.Country = df['Country'].map(countryMap)

    df = df[df['Salary'] <= 250000]
    df = df[df['Salary'] > 10000]
    df = df[df['Country'] != 'Other']

    df['YearsCodePro'] = df['YearsCodePro'].apply(setExperience)
    df['EdLevel'] = df['EdLevel'].apply(setEducation)
    return df

df = load_data()

def dataDist():
    viz = df['Country'].value_counts()
    fig1,ax1 = plt.subplots()
    ax1.pie(viz,labels=viz.index,autopct='%1.1f%%',shadow=True,startangle=90)
    ax1.axis('equal')
    st.subheader("""
    Data Distribution by Countries
    """)
    st.pyplot(fig1)

    st.subheader("""
    Mean Salary by Countries
    
    """)
    data = df.groupby(['Country'])['Salary'].mean().sort_values(ascending=True)
    st.bar_chart(data)

    ##line chart
    st.subheader("""
        Mean Salary by Experience

        """)
    data = df.groupby(['YearsCodePro'])['Salary'].mean().sort_values(ascending=True)
    st.line_chart(data)



dist = dataDist()

