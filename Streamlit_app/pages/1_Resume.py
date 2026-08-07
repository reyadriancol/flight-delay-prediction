# =========================================================================
# AI USAGE CITATION
#
# Tool:   Anthropic Claude (Opus 5), July 2026
#
# Prompt: Requests for help setting up the layout and structure of a
#         Streamlit resume page: page configuration, the profile photo
#         and heading arrangement, and the Streamlit components used to
#         organize sections (columns, headers, markdown blocks).
#
# Usage:  AI was used for the structure and Streamlit scaffolding of this
#         page only. All resume content, including the professional
#         summary, education history, technical skills, and experience
#         bullets, was written by the author. No AI-generated text
#         appears in the content of this page. All AI-suggested layout
#         code was reviewed and adapted by the author.
# =========================================================================

from pathlib import Path

import streamlit as st
from PIL import Image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Resume | Rey Colongon",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# File paths
# --------------------------------------------------
APP_FOLDER = Path(__file__).resolve().parent.parent
PHOTO_PATH = APP_FOLDER / "portfolio_pic.jpg"


# --------------------------------------------------
# Resume heading
# --------------------------------------------------
left_column, right_column = st.columns([1, 3], vertical_alignment="center")

with left_column:
    if PHOTO_PATH.exists():
        profile_image = Image.open(PHOTO_PATH)

        st.image(
            profile_image,
            width=220
        )
    else:
        st.info(
            "Add your photo as "
            "`Streamlit_app/portfolio_pic.jpg`."
        )

with right_column:
    st.title("Rey Colongon")

    st.subheader(
        "Data & Analytics Professional"
    )

    st.write("📍 Uruma City, Okinawa, Japan 904-1111")

    st.markdown(
        """
                ✉️ [reyadrian.colongon@gmail.com](mailto:reyadrian.colongon@gmail.com)
        """
    )


st.divider()


# --------------------------------------------------
# Professional summary
# --------------------------------------------------
st.header("Professional Summary")

st.write("""
Analytics-focused professional with a B.S. in Business Analytics and an
in-progress M.S. in Data Science, backed by more than 14 years of experience
supporting mission-critical defense operations in Okinawa, Japan.

Skilled in analyzing operational and maintenance data to diagnose problems,
identify recurring patterns, and support process improvements that increase
equipment reliability and reduce downtime. Strengths include statistical
reasoning, data quality, systematic troubleshooting, and communicating
technical findings to cross-functional teams.

Former U.S. Air Force technician with a record of independent ownership,
quality-focused work, and zero-defect inspection performance.
""")


st.divider()


# --------------------------------------------------
# Education
# --------------------------------------------------
st.header("Education")

st.subheader("Master of Science in Data Science")
st.write("**Eastern University**")
st.write("In Progress • Expected August 2026")

st.markdown("""
**Areas of Study**

- Machine learning
- Statistical analysis
- Python programming
- Data cleaning and wrangling
- Exploratory data analysis
- Predictive modeling
""")

st.subheader("Bachelor of Science in Business Analytics")
st.write("**Embry-Riddle Aeronautical University**")
st.write("Cum Laude • 2025")

st.subheader("Aerospace Ground Equipment Technical Certification")
st.write("**United States Air Force**")
st.write("2011")


st.divider()


# --------------------------------------------------
# Technical skills
# --------------------------------------------------
st.header("Technical Skills")

skill_column1, skill_column2 = st.columns(2)

with skill_column1:
    st.subheader("Programming & Querying")

    st.markdown("""
- Python
- R
- SQL
""")

    st.subheader("Analytics & Modeling")

    st.markdown("""
- Machine learning with scikit-learn
- Statistical analysis
- Data cleaning and wrangling
- Exploratory data analysis
- Feature engineering
- Model evaluation
""")

with skill_column2:
    st.subheader("Data & Visualization")

    st.markdown("""
- pandas
- Jupyter Notebook
- Streamlit
- Tableau
- Microsoft Excel
- Operational data analysis
""")

    st.subheader("Professional Strengths")

    st.markdown("""
- Data quality and compliance
- Process improvement
- Maintenance data analysis
- Technical troubleshooting
- Cross-functional communication
- Quality assurance
""")


st.divider()


# --------------------------------------------------
# Professional experience
# --------------------------------------------------
st.header("Professional Experience")

st.subheader(
    "Product Repair & Modification Technician • "
    "Collateral Duty Inspector (CDI)"
)

st.write("**The Boeing Company — Okinawa, Japan**")
st.write("*January 2019 – Present*")

st.markdown("""
- Analyze technical and maintenance data in REDARS to diagnose equipment
  issues, identify recurring failure patterns, and prioritize repairs for
  mission-critical ground support equipment.

- Review maintenance history and operational conditions to identify recurring
  equipment problems and support process improvements that reduce repeat
  failures and increase equipment availability.

- Independently inspect and certify completed repairs against technical
  specifications as a Collateral Duty Inspector, ensuring compliance with
  Boeing and U.S. Navy safety and quality requirements.

- Safeguard the accuracy and integrity of maintenance records through detailed
  inspections, documentation reviews, and quality-control procedures.

- Communicate technical findings and repair decisions across engineering,
  operations, customer, and maintenance teams.
""")


st.subheader("Aerospace Ground Equipment Technician")

st.write("**United States Air Force — Multiple Locations**")
st.write("*October 2010 – December 2018*")

st.markdown("""
- Led a comprehensive review of inventory and bench-stock data, resolved
  long-standing discrepancies, and supported full data compliance during a
  subsequent high-level Air Force inspection.

- Diagnosed and maintained electro-mechanical support equipment used for more
  than 100 F-16 and F-35 aircraft.

- Performed systematic troubleshooting of complex electrical, mechanical,
  hydraulic, pneumatic, and environmental-control systems.

- Earned selection to the Air Force Quality Assurance Honor Roll after passing
  seven inspections with zero defects.

- Received a Commander's Coin for performance, leadership, and attention to
  detail.

- Trained personnel in quality-focused maintenance and inspection techniques
  and contributed to the unit's recognition for maintenance effectiveness.
""")


st.divider()

