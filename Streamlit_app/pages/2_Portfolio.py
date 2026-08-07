# =========================================================================
# AI USAGE CITATION
#
# Tool:   Anthropic Claude (Opus 5), July 2026
#
# Prompt: Requests for help setting up the layout and structure of a
#         Streamlit portfolio page: page configuration, section
#         organization, and the Streamlit components used to arrange
#         content (columns, headers, metrics, markdown blocks).
#
# Usage:  AI was used for the structure and Streamlit scaffolding of this
#         page only. All written content, including the project
#         description, problem statement, workflow steps, feature list,
#         model list, tools, skills, and closing reflection, was written
#         by the author. No AI-generated text appears in the content of
#         this page. All AI-suggested layout code was reviewed and
#         adapted by the author.
# =========================================================================


import streamlit as st

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Portfolio",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio")

st.write("""
Welcome to my project portfolio. This page highlights some of the data science
and analytics projects I have completed during my academic studies.

My projects focus on data cleaning, exploratory data analysis, machine learning,
model evaluation, and creating interactive applications.
""")

st.divider()

# --------------------------------------------------
# Project 1: Flight Delay Prediction
# --------------------------------------------------
st.header("✈️ Flight Delay Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Project Overview")

    st.write("""
This project predicts the probability that a scheduled U.S. domestic flight
will arrive 15 or more minutes late.

The model uses only information available before departure, including the
airline, origin airport, destination airport, scheduled departure time,
flight duration, day of the week, month, and proximity to a U.S. holiday.
""")

    st.subheader("Problem")

    st.write("""
Flight delays can affect passengers, airlines, airports, and transportation
planning. The goal of this project was to build a machine learning model
that estimates delay risk without using information that would only become
available after the flight begins.
""")

with col2:
    st.metric("Dataset Size", "6.8M+ Flights")
    st.metric("Prediction Target", "15+ Minute Delay")

st.subheader("Data Source")

st.write("""
The project uses the 2025 U.S. Department of Transportation Bureau of
Transportation Statistics On-Time Performance dataset. The twelve monthly
datasets were combined into a single dataset containing over 6.8 million
domestic flights.
""")

st.subheader("Project Workflow")

st.markdown("""
1. Imported and combined twelve monthly datasets.
2. Removed cancelled and diverted flights.
3. Cleaned missing values.
4. Performed exploratory data analysis (EDA).
5. Engineered new time and holiday features.
6. Split the data into training, validation, and testing sets.
7. Trained multiple machine learning models.
8. Evaluated models using Precision, Recall, F1 Score, and ROC-AUC.
9. Deployed the final model using Streamlit.
""")

st.subheader("Features Used")

left, right = st.columns(2)

with left:
    st.markdown("""
- Airline
- Origin Airport
- Destination Airport
- Month
""")

with right:
    st.markdown("""
- Day of Week
- Departure Hour
- Scheduled Flight Duration
- Holiday Proximity
""")

st.subheader("Models Evaluated")

st.write("""
The project compared several classification algorithms including:

- Logistic Regression
- Random Forest
- SGD Logistic Regression
- Histogram Gradient Boosting
""")

st.subheader("Tools & Technologies")

st.write("""
**Programming:** Python

**Libraries:** pandas, NumPy, scikit-learn, Matplotlib

**Development Environment:** Jupyter Notebook, Anaconda
""")

st.subheader("Key Skills Demonstrated")

st.markdown("""
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning
- Model Evaluation
- Classification
- Predictive Analytics
- Streamlit Deployment
""")

with st.container(border=True):
    st.subheader("Flight Delay Prediction")
    st.write("Predicts the probability that a U.S. domestic flight arrives 15+ minutes late.")
    st.page_link(
        "pages/3_Flight_Delay_Simulator.py",
        label="Try the simulator",
        icon="✈️"
    )

st.divider()

# --------------------------------------------------
# Closing
# --------------------------------------------------

st.header("What I Learned")

st.write("""
This project strengthened my understanding of the complete data science
workflow. I gained practical experience working with a large dataset,
performing feature engineering, comparing machine learning models,
interpreting evaluation metrics, and deploying a predictive model
as an interactive Streamlit application.
""")