import streamlit as st
st.title('Software Developer Salary Prediction')

st.write("""
>This Machine Predict Salary of a Software Engineer using various prediction like, **Experience** and **Education**.
""")
st.subheader('Explore or Predict')
explore = st.markdown("""
-**Model Prediction** [Predict](http://localhost:8501/prediction),
-**Data Exploration** [Explore](http://localhost:8501/EDA)

""")