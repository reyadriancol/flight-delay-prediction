# =========================================================================
# AI USAGE CITATION
#
# Tool:   Anthropic Claude (Opus 5), July 2026
#
# Prompt: Requests for help setting up the layout and structure of a
#         Streamlit biographical homepage: page configuration, section
#         organization, and the Streamlit components used to arrange
#         content.
#
# Usage:  AI was used for the structure and Streamlit scaffolding of this
#         page only. All written content was written by the author. No
#         AI-generated text appears in the content of this page. All
#         AI-suggested layout code was reviewed and adapted by the
#         author.
# =========================================================================

import streamlit as st

st.title("Rey Colongon")
st.subheader("MS in Data Science Candidate — Eastern University")
st.divider()

st.header("About Me")
st.write("""
My name is Rey Colongon, and I hold a Master of Science in Data Science from Eastern University. My interest in data science comes from enjoying problem-solving and discovering meaningful patterns in data. I enjoy building predictive models, learning new technologies, and continuously improving my analytical and programming skills.

I have lived and worked in Okinawa, Japan for seven years, supporting aviation maintenance operations. I live here with my wife and our two-year-old daughter. Outside of work and school, I am into coffee and cooking. Living in Japan has changed how I cook quite a bit. We also travel when we can, which is easier from where we are than most places.

I am looking to move into a role where analysis is the actual job: data analysis, machine learning, or predictive modeling, ideally somewhere the results feed real decisions. My aviation and maintenance background gives me a domain I already know well.""")

st.header("Academic Background")
st.write("""
I earned my Bachelor of Science in Business Analytics (cum laude) from Embry-Riddle
Aeronautical University and and my Master of Science in Data Science (GPA 3.867) from Eastern University. 
My coursework has included machine learning, statistics, programming with Python, data visualization, 
and predictive analytics. Through my projects, I have gained experience with data cleaning,
exploratory data analysis, feature engineering, and model evaluation.
""")

st.header("Career Aspirations")
st.write("""
My goal is to become a data scientist where I can use data to solve real-world
business problems and support data-driven decision making. I hope to continue
developing my machine learning skills while working on projects that have a
meaningful impact. I am particularly interested in opportunities that allow me
to combine technical expertise with business strategy.
""")

st.header("Professional Interests")
st.write("""
I am most interested in machine learning, predictive analytics, and data
visualization. I enjoy exploring how models can identify patterns and generate
insights from large datasets. I am also interested in cloud technologies,
model deployment, and learning best practices for building scalable data
science solutions.
""")