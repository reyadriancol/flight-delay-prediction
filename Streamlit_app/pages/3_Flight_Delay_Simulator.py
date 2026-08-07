# =========================================================================
# AI USAGE CITATION
#
# Tool:   Anthropic Claude (Opus 5), July 2026
#
# Prompt: Summary of prompts used across development of this file --
#         requests to build the Streamlit prediction page, wire in the
#         route_duration lookup so scheduled duration is derived rather
#         than entered, display exact days to the nearest holiday, colour
#         the risk bar from blue to red, and add carrier comparison and
#         day-by-hour heatmap views. Also submitted an earlier
#         author-written draft of this page for review.
#
# Usage:  AI was used substantially throughout this file. The author wrote
#         the original page structure; AI reviewed it and generated most
#         of the code in its current form. All model development, feature
#         engineering, and evaluation decisions were made independently by
#         the author, and all AI-generated code was reviewed, configured,
#         and tested by the author before use.
# =========================================================================

from datetime import date
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
import holidays


# =========================================================================
# Page configuration
# =========================================================================
st.set_page_config(
    page_title="Flight Delay Simulator | Rey Colongon",
    page_icon="✈️",
    layout="wide"
)


# =========================================================================
# File paths
#
# This file lives in pages/, so the model and the lookup table are one
# level up in the main Streamlit_app folder.
# =========================================================================
APP_FOLDER = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_FOLDER / "best_hgb_model.pkl"
ROUTE_PATH = APP_FOLDER / "route_duration.csv"

HOLIDAY_CAP = 7   # must match the cap used in the training notebook

DAY_NAMES = {
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
    7: "Sunday"
}


# =========================================================================
# Cached loaders
#
# Streamlit re-runs this entire script on every widget interaction. Caching
# means the model and the lookup table are read from disk once per session
# instead of on every click.
# =========================================================================
@st.cache_resource
def load_model(model_path):
    """Load the trained pipeline.

    joblib.load reads files written by both joblib.dump and pickle.dump,
    so the pickle fallback is only for unusual cases.
    """
    try:
        return joblib.load(model_path)
    except Exception:
        import pickle
        with open(model_path, "rb") as file_handle:
            return pickle.load(file_handle)


@st.cache_data
def load_routes(route_path):
    """Load the median scheduled duration for each ORIGIN to DEST pair.

    This table was built in the notebook from the cleaned training data:
        df.groupby(['ORIGIN', 'DEST'])['CRS_ELAPSED_TIME'].median()

    It serves two purposes. It supplies the duration the user cannot
    reasonably be expected to know, and it defines which airports and
    routes are selectable, so the model is never asked to score a route
    it did not see during training.
    """
    return pd.read_csv(route_path)


# =========================================================================
# Holiday proximity
#
# get_holiday_proximity must reproduce the notebook function exactly, since
# its output is fed to the model. If the notebook used pandas
# USFederalHolidayCalendar rather than the holidays package, replace this
# body to match, because the two libraries disagree on observed dates for
# holidays that fall on a weekend.
#
# get_nearest_holiday is display only. It reports the true distance and the
# holiday's name, and is never passed to the model.
# =========================================================================
def _holiday_window(flight_date):
    """US federal holidays across a three-year window.

    The window spans the previous through the following year so that dates
    in early January measure back to the previous New Year's Day and dates
    in late December measure forward to the next one.
    """
    flight_date = pd.Timestamp(flight_date).normalize()

    years = [
        flight_date.year - 1,
        flight_date.year,
        flight_date.year + 1
    ]

    return flight_date, holidays.US(years=years)


def get_holiday_proximity(flight_date, cap=HOLIDAY_CAP):
    """Capped days to the nearest holiday. This is the model feature."""
    flight_date, us_holidays = _holiday_window(flight_date)

    minimum_distance = min(
        abs((flight_date - pd.Timestamp(holiday_date)).days)
        for holiday_date in us_holidays.keys()
    )

    return min(minimum_distance, cap)


def get_nearest_holiday(flight_date):
    """Exact days to the nearest holiday and its name. Display only."""
    flight_date, us_holidays = _holiday_window(flight_date)

    nearest_date, nearest_name = min(
        us_holidays.items(),
        key=lambda item: abs((flight_date - pd.Timestamp(item[0])).days)
    )

    exact_days = abs((flight_date - pd.Timestamp(nearest_date)).days)

    return exact_days, nearest_name


# =========================================================================
# Risk presentation
#
# The model was trained with class_weight='balanced', which reweights the
# training distribution so delays appear as common as on-time arrivals.
# Predicted probabilities are therefore calibrated to that reweighted
# world, not the real one, and cluster around 0.5. They are reliable for
# ranking one flight against another but are not literal historical
# frequencies, so the output is presented as a comparative risk score.
# =========================================================================
def get_risk_category(probability):
    """Translate a model score into a plain-language risk band."""
    if probability < 0.40:
        return "Lower", "This flight scores below average for delay risk."

    if probability < 0.55:
        return "Typical", "This flight scores near average for delay risk."

    if probability < 0.70:
        return "Elevated", "This flight scores above average for delay risk."

    return "High", "This flight scores well above average for delay risk."


# Colour stops for the risk meter, running cool to warm. Each entry is a
# position between 0 and 1 paired with an RGB triple.
HEAT_STOPS = [
    (0.00, (37, 99, 235)),     # blue
    (0.35, (14, 165, 233)),    # cyan
    (0.50, (234, 179, 8)),     # amber
    (0.70, (249, 115, 22)),    # orange
    (1.00, (220, 38, 38))      # red
]


def get_heat_color(probability):
    """Interpolate a hex colour along the blue to red ramp.

    The score is placed between the two surrounding stops and each RGB
    channel is blended proportionally, so the bar shifts smoothly rather
    than jumping between fixed colours.
    """
    probability = max(0.0, min(1.0, float(probability)))

    for index in range(len(HEAT_STOPS) - 1):
        lower_position, lower_rgb = HEAT_STOPS[index]
        upper_position, upper_rgb = HEAT_STOPS[index + 1]

        if lower_position <= probability <= upper_position:
            span = upper_position - lower_position
            weight = 0.0 if span == 0 else (probability - lower_position) / span

            blended = [
                round(lower_rgb[channel]
                      + (upper_rgb[channel] - lower_rgb[channel]) * weight)
                for channel in range(3)
            ]

            return "#{:02X}{:02X}{:02X}".format(*blended)

    return "#{:02X}{:02X}{:02X}".format(*HEAT_STOPS[-1][1])


def render_risk_meter(probability):
    """Draw a coloured risk bar using inline HTML.

    Streamlit's built-in st.progress cannot be recoloured, so the bar is
    drawn directly. The track uses a translucent grey so it reads correctly
    in both the light and dark themes.
    """
    fill_color = get_heat_color(probability)
    fill_percent = probability * 100

    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 0.25rem 0;">
          <div style="
              width: 100%;
              height: 26px;
              background: rgba(128, 128, 128, 0.18);
              border-radius: 13px;
              overflow: hidden;
          ">
            <div style="
                width: {fill_percent:.1f}%;
                height: 100%;
                background: {fill_color};
                border-radius: 13px;
            "></div>
          </div>
          <div style="
              display: flex;
              justify-content: space-between;
              font-size: 0.75rem;
              opacity: 0.7;
              margin-top: 0.25rem;
          ">
            <span>Lower risk</span>
            <span><strong>{probability:.1%}</strong></span>
            <span>Higher risk</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================================
# Airline choices
#
# Codes must match the values of OP_UNIQUE_CARRIER in the training data.
# Verify with sorted(df['OP_UNIQUE_CARRIER'].unique()) in the notebook.
# =========================================================================
AIRLINE_OPTIONS = {
    "American Airlines": "AA",
    "Alaska Airlines": "AS",
    "JetBlue Airways": "B6",
    "Delta Air Lines": "DL",
    "Frontier Airlines": "F9",
    "Allegiant Air": "G4",
    "Hawaiian Airlines": "HA",
    "Spirit Airlines": "NK",
    "SkyWest Airlines": "OO",
    "United Airlines": "UA",
    "Southwest Airlines": "WN",
    "Republic Airways": "YX"
}

CODE_TO_AIRLINE = {code: name for name, code in AIRLINE_OPTIONS.items()}


# =========================================================================
# Scoring helper
#
# Every comparison view works the same way: copy the base feature row,
# vary one or two fields, and score the whole batch in a single call. The
# pipeline handles a many-row frame just as easily as a one-row frame, so
# 168 scenarios cost roughly the same as one.
# =========================================================================
def score_variants(model, base_features, expected_columns, overrides):
    """Score a list of variations on the base flight.

    overrides is a list of dictionaries, each holding the fields to change
    for that scenario. Returns the predicted delay score for each.
    """
    rows = []

    for override in overrides:
        row = dict(base_features)
        row.update(override)
        rows.append(row)

    frame = pd.DataFrame(rows)[expected_columns]

    return model.predict_proba(frame)[:, 1]


# =========================================================================
# Page heading
# =========================================================================
st.title("✈️ Flight Delay Prediction Simulator")

st.write(
    """
    Enter the scheduled flight information below to estimate the risk that
    the flight will arrive **15 minutes or more late**.
    """
)

st.info(
    """
    This tool provides a model-based estimate, not a guarantee. It uses only
    information known at the time a flight is scheduled. Actual delays are
    also affected by weather, air traffic, maintenance, and other real-time
    conditions the model cannot see.
    """
)


# =========================================================================
# Confirm required files exist before doing anything else
# =========================================================================
missing_files = [
    path.name for path in (MODEL_PATH, ROUTE_PATH) if not path.exists()
]

if missing_files:
    st.error(
        "Required file(s) not found: "
        + ", ".join(missing_files)
        + ". Place them in the main Streamlit_app folder."
    )

    st.code(
        """
Streamlit_app/
├── Home.py
├── best_hgb_model.pkl
├── route_duration.csv
└── pages/
    └── 3_Flight_Delay_Simulator.py
        """,
        language="text"
    )

    st.stop()


# =========================================================================
# Load model and lookup table
# =========================================================================
try:
    model = load_model(MODEL_PATH)
    routes = load_routes(ROUTE_PATH)

except Exception as error:
    st.error("The model or route table could not be loaded.")
    st.exception(error)

    st.write(
        """
        Make sure the model was saved using compatible versions of Python,
        scikit-learn, pandas, NumPy, and joblib.
        """
    )

    st.stop()


# =========================================================================
# Route selection
#
# The origin must be read before the form is submitted so the destination
# list can be filtered to it, so these controls sit outside the form.
# =========================================================================
st.header("Flight Information")

route_column1, route_column2 = st.columns(2)

with route_column1:
    origin = st.selectbox(
        "Origin airport",
        options=sorted(routes["ORIGIN"].unique()),
        index=None,
        placeholder="Select an origin airport",
        help="Only airports present in the training data are listed."
    )

with route_column2:
    if origin is None:
        destination = st.selectbox(
            "Destination airport",
            options=[],
            index=None,
            placeholder="Select an origin first",
            disabled=True
        )
    else:
        available_destinations = sorted(
            routes.loc[routes["ORIGIN"] == origin, "DEST"].unique()
        )

        destination = st.selectbox(
            "Destination airport",
            options=available_destinations,
            index=None,
            placeholder="Select a destination airport",
            help=(
                f"{len(available_destinations)} destinations were served "
                f"from {origin} in the training data."
            )
        )

if origin is None or destination is None:
    st.info("Select an origin and destination to continue.")
    st.stop()


# The duration is looked up rather than entered. A traveller booking a
# flight does not know its scheduled elapsed time, but the route implies
# it, so the value is filled automatically from the training data.
elapsed_time = float(
    routes.loc[
        (routes["ORIGIN"] == origin) & (routes["DEST"] == destination),
        "CRS_ELAPSED_TIME"
    ].iloc[0]
)

st.caption(
    f"Typical scheduled duration for {origin} to {destination}: "
    f"{elapsed_time:.0f} minutes. This value is taken from the training "
    f"data and is not entered by the user."
)


# =========================================================================
# Remaining inputs
# =========================================================================
with st.form("flight_prediction_form"):

    date_column, airline_column = st.columns(2)

    with date_column:
        flight_date = st.date_input(
            "Scheduled flight date",
            value=date.today(),
            help=(
                "Month, day of the week, and holiday proximity are "
                "calculated automatically from this date."
            )
        )

    with airline_column:
        airline_name = st.selectbox(
            "Operating airline",
            options=list(AIRLINE_OPTIONS.keys())
        )

        airline_code = AIRLINE_OPTIONS[airline_name]

    departure_hour = st.slider(
        "Scheduled departure hour",
        min_value=0,
        max_value=23,
        value=12,
        step=1,
        help="Use 0 for midnight, 12 for noon, and 23 for 11:00 PM."
    )

    predict_button = st.form_submit_button(
        "Estimate Delay Risk",
        type="primary",
        use_container_width=True
    )


# =========================================================================
# Prediction
# =========================================================================
if predict_button:

    # The training data encodes DAY_OF_WEEK as Monday=1 through Sunday=7.
    # Python's weekday() returns Monday=0, so one is added.
    # Verify against df['DAY_OF_WEEK'].unique() in the notebook.
    month = flight_date.month
    day_of_week = flight_date.weekday() + 1

    holiday_proximity = get_holiday_proximity(flight_date, cap=HOLIDAY_CAP)
    exact_days, holiday_name = get_nearest_holiday(flight_date)

    feature_values = {
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "OP_UNIQUE_CARRIER": airline_code,
        "ORIGIN": origin,
        "DEST": destination,
        "CRS_ELAPSED_TIME": elapsed_time,
        "DEP_HOUR": int(departure_hour),
        "HOLIDAY_PROX": int(holiday_proximity)
    }

    # feature_names_in_ is recorded when the pipeline is fitted. Building
    # the row against it guarantees the column names, order, and count
    # match training exactly, instead of relying on a hand-typed list that
    # can drift out of sync with the notebook.
    expected_columns = list(
        getattr(model, "feature_names_in_", feature_values.keys())
    )

    missing_columns = [
        column for column in expected_columns
        if column not in feature_values
    ]

    if missing_columns:
        st.error(
            "The trained model expects columns this page does not build: "
            f"{missing_columns}"
        )
        st.stop()

    flight_input = pd.DataFrame([feature_values])[expected_columns]

    try:
        probabilities = model.predict_proba(flight_input)
        delay_probability = float(probabilities[0][1])

    except Exception as error:
        st.error("The prediction could not be completed.")
        st.exception(error)

        st.write("The information sent to the model was:")
        st.dataframe(flight_input, use_container_width=True, hide_index=True)
        st.stop()

    # ---------------------------------------------------------------------
    # Results
    #
    # Only the model score and its risk band are shown. A hard delayed or
    # not-delayed verdict is deliberately omitted: predict() applies a
    # fixed 0.5 cutoff, which on a class-balanced model fires frequently
    # and would contradict the score displayed beside it.
    # ---------------------------------------------------------------------
    st.divider()
    st.header("Results")

    risk_category, risk_description = get_risk_category(delay_probability)

    result_column1, result_column2 = st.columns(2)

    with result_column1:
        st.metric("Model delay score", f"{delay_probability:.1%}")

    with result_column2:
        st.metric("Risk level", risk_category)

    render_risk_meter(delay_probability)

    st.write(risk_description)

    st.caption(
        "This score is a comparative measure, not a historical frequency. "
        "The model was trained with balanced class weights so that delays "
        "were not overwhelmed by on-time flights, which shifts the scores "
        "upward. A score of 60 percent means this flight ranks as higher "
        "risk than one scoring 40 percent, not that six in ten such "
        "flights arrive late."
    )

    # ---------------------------------------------------------------------
    # Flight summary
    # ---------------------------------------------------------------------
    st.subheader("Flight Summary")

    summary_column1, summary_column2 = st.columns(2)

    with summary_column1:
        st.write(f"**Route:** {origin} → {destination}")
        st.write(f"**Airline:** {airline_name} ({airline_code})")
        st.write(f"**Flight date:** {flight_date.strftime('%B %d, %Y')}")
        st.write(f"**Month used by model:** {month}")

    with summary_column2:
        st.write(f"**Day of week used by model:** {day_of_week}")
        st.write(f"**Departure hour:** {departure_hour}:00")
        st.write(f"**Scheduled duration:** {elapsed_time:.0f} minutes")

        if exact_days == 0:
            st.write(
                f"**Nearest U.S. holiday:** {holiday_name} "
                f"(the flight is on it)"
            )
        else:
            st.write(
                f"**Nearest U.S. holiday:** {holiday_name}, "
                f"{exact_days} day{'s' if exact_days != 1 else ''} away"
            )

        st.write(
            f"**Holiday proximity value used by model:** "
            f"{holiday_proximity}"
        )

    if exact_days > HOLIDAY_CAP:
        st.caption(
            f"The holiday proximity feature is capped at {HOLIDAY_CAP} days. "
            f"Beyond roughly a week, additional distance from a holiday "
            f"carries no further signal, so every such date is treated "
            f"identically by the model."
        )

    with st.expander("View the exact information sent to the model"):
        st.dataframe(flight_input, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------
    # Carrier comparison
    #
    # The same route, date, and departure hour are re-scored for every
    # airline. This shows how much of the estimate is driven by the choice
    # of carrier rather than by the route or the schedule.
    # ---------------------------------------------------------------------
    st.divider()
    st.header("How Other Airlines Compare")

    st.write(
        f"""
        Every airline scored on the same route, date, and departure hour as
        the flight above. Lower is better.
        """
    )

    carrier_codes = list(AIRLINE_OPTIONS.values())

    carrier_scores = score_variants(
        model,
        feature_values,
        expected_columns,
        [{"OP_UNIQUE_CARRIER": code} for code in carrier_codes]
    )

    carrier_frame = pd.DataFrame({
        "Airline": [CODE_TO_AIRLINE[code] for code in carrier_codes],
        "Code": carrier_codes,
        "Score": carrier_scores
    }).sort_values("Score").reset_index(drop=True)

    best_row = carrier_frame.iloc[0]
    worst_row = carrier_frame.iloc[-1]

    st.write(
        f"On this route the model scores **{best_row['Airline']}** lowest "
        f"at {best_row['Score']:.1%} and **{worst_row['Airline']}** highest "
        f"at {worst_row['Score']:.1%}, a spread of "
        f"{(worst_row['Score'] - best_row['Score']):.1%}."
    )

    carrier_chart = (
        alt.Chart(carrier_frame)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("Score:Q", title="Model delay score",
                    axis=alt.Axis(format="%")),
            y=alt.Y("Airline:N", sort="x", title=None),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(scheme="redyellowblue", reverse=True),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("Airline:N"),
                alt.Tooltip("Score:Q", format=".1%", title="Delay score")
            ]
        )
        .properties(height=380)
    )

    st.altair_chart(carrier_chart, use_container_width=True)

    st.caption(
        "The model will score any airline on any route, including "
        "combinations that no carrier actually operates. Treat an airline "
        "that does not serve this route as hypothetical. Differences here "
        "reflect each carrier's overall on-time pattern in the training "
        "data, not a causal claim about that airline."
    )

    # ---------------------------------------------------------------------
    # Day and hour heatmap
    #
    # 168 scenarios, one per combination of weekday and departure hour,
    # scored in a single batched call. This makes the schedule structure
    # the model learned directly visible.
    # ---------------------------------------------------------------------
    st.divider()
    st.header("Best Times to Fly This Route")

    st.write(
        f"""
        Every combination of day and departure hour for {origin} to
        {destination} on {airline_name} in
        {flight_date.strftime('%B')}. Cooler cells score lower.
        """
    )

    grid_overrides = [
        {"DAY_OF_WEEK": day, "DEP_HOUR": hour}
        for day in range(1, 8)
        for hour in range(24)
    ]

    grid_scores = score_variants(
        model, feature_values, expected_columns, grid_overrides
    )

    grid_frame = pd.DataFrame({
        "Day": [DAY_NAMES[item["DAY_OF_WEEK"]] for item in grid_overrides],
        "Hour": [item["DEP_HOUR"] for item in grid_overrides],
        "Score": grid_scores
    })

    heatmap = (
        alt.Chart(grid_frame)
        .mark_rect()
        .encode(
            x=alt.X("Hour:O", title="Scheduled departure hour"),
            y=alt.Y("Day:N", title=None,
                    sort=[DAY_NAMES[number] for number in range(1, 8)]),
            color=alt.Color(
                "Score:Q",
                scale=alt.Scale(scheme="redyellowblue", reverse=True),
                legend=alt.Legend(title="Delay score", format=".0%")
            ),
            tooltip=[
                alt.Tooltip("Day:N"),
                alt.Tooltip("Hour:O", title="Departure hour"),
                alt.Tooltip("Score:Q", format=".1%", title="Delay score")
            ]
        )
        .properties(height=300)
    )

    st.altair_chart(heatmap, use_container_width=True)

    best_slot = grid_frame.loc[grid_frame["Score"].idxmin()]
    worst_slot = grid_frame.loc[grid_frame["Score"].idxmax()]

    st.write(
        f"The lowest-scoring slot is **{best_slot['Day']} at "
        f"{int(best_slot['Hour'])}:00** ({best_slot['Score']:.1%}). "
        f"The highest is **{worst_slot['Day']} at "
        f"{int(worst_slot['Hour'])}:00** ({worst_slot['Score']:.1%})."
    )

    st.caption(
        "These cells show how the score moves as the departure day and "
        "hour change, holding everything else fixed. That is a sensitivity "
        "view, not a statement about what causes delays. Very early and "
        "very late slots may also reflect thin scheduling in the training "
        "data rather than genuinely different operating conditions."
    )


# =========================================================================
# Model explanation
# =========================================================================
st.divider()
st.header("How the Model Works")

st.write(
    """
    The model evaluates patterns from historical U.S. domestic flights during
    2025. It combines route, airline, schedule, flight duration, departure
    time, and holiday proximity to estimate the risk of an arrival delay of
    at least 15 minutes.
    """
)

explanation_column1, explanation_column2 = st.columns(2)

with explanation_column1:
    st.subheader("Inputs Used")

    st.markdown(
        """
        - Flight month
        - Day of the week
        - Operating airline
        - Origin airport
        - Destination airport
        - Scheduled flight duration
        - Scheduled departure hour
        - Proximity to a U.S. federal holiday
        """
    )

with explanation_column2:
    st.subheader("Important Limitation")

    st.markdown(
        """
        The model does not use real-time information such as:

        - Current weather
        - Aircraft maintenance events
        - Air traffic restrictions
        - Crew availability
        - Previous-flight delays
        - Airport emergencies
        """
    )

st.subheader("How Well Does It Work?")

st.write("""
The model was tested on flights it had never seen during training. Of the
flights that actually arrived late, it correctly flagged about 64 percent
in advance.

For comparison, a simple approach that predicts the overall average delay
rate for every flight identifies none of them, because it treats all
flights as identical. The model's value is in telling flights apart.

It is not always right. Roughly a third of the flights it flags as higher
risk arrive on time. That trade is deliberate: the tool is tuned to catch
as many real delays as possible, accepting more false alarms in exchange,
because a missed delay costs a traveler more than an unnecessary warning.
""")