
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Diabetes Check",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 92% 0%, rgba(99,102,241,.10), transparent 28%),
            radial-gradient(circle at 0% 90%, rgba(14,165,233,.08), transparent 25%),
            #f7f9fc;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 28px;
        padding-bottom: 35px;
    }

    [data-testid="stSidebar"] {
        background: #101827;
    }

    [data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 26px;
        color: white;
        background: linear-gradient(135deg, #0f172a 0%, #332c83 55%, #0f766e 100%);
        box-shadow: 0 18px 42px rgba(15,23,42,.15);
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        line-height: 1.1;
        font-weight: 800;
        letter-spacing: -.7px;
    }

    .hero p {
        margin: 9px 0 0;
        font-size: 15px;
        opacity: .92;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(15,23,42,.05);
    }

    .section-title {
        color: #0f172a;
        font-size: 24px;
        font-weight: 800;
        margin: 12px 0 4px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .guide {
        background: linear-gradient(135deg, #eef2ff 0%, #f0fdfa 100%);
        border: 1px solid #c7d2fe;
        border-radius: 17px;
        padding: 17px 18px;
        color: #1e293b;
        margin-bottom: 20px;
        line-height: 1.55;
    }

    .guide strong {
        color: #312e81;
    }

    /* IMPORTANT: keep Streamlit labels visible */
    [data-testid="stWidgetLabel"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin-bottom: 4px !important;
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #111827 !important;
        font-size: 14px !important;
        font-weight: 750 !important;
    }

    /* Select / number controls */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        min-height: 44px !important;
    }

    div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }

    div[data-baseweb="input"] input {
        color: #111827 !important;
        background: #ffffff !important;
    }

    /* Slider */
    div[data-testid="stSlider"] {
        padding-top: 2px !important;
        padding-bottom: 8px !important;
    }

    /* Buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        min-height: 48px;
        border-radius: 12px;
        font-size: 15px;
        font-weight: 800;
        border: 0;
    }

    .result {
        padding: 26px;
        border-radius: 20px;
        margin-top: 22px;
        text-align: center;
    }

    .result h2 {
        margin: 0;
        font-size: 29px;
        font-weight: 850;
    }

    .result p {
        margin: 9px 0 0;
        font-size: 14px;
    }

    .result-good {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
    }

    .result-review {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 18px 20px;
        box-shadow: 0 8px 23px rgba(15,23,42,.05);
    }

    .stat-label {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
    }

    .stat-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 850;
        margin-top: 4px;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        font-size: 12px;
        padding-top: 28px;
    }

    /* Make slider values easy to read */
    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-left: 2px !important;
        padding-right: 2px !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE / MODEL SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATHS = [
    BASE_DIR / "data" / "Dataset of Diabetes.csv",
    BASE_DIR / "Dataset of Diabetes.csv",
    BASE_DIR / "Dataset of Diabetes .csv",
]

FEATURES = [
    "AGE",
    "Urea",
    "Cr",
    "HbA1c",
    "Chol",
    "TG",
    "HDL",
    "LDL",
    "VLDL",
    "BMI",
    "Gender_M",
]

# Starting demo profile.
# These are example inputs only, not a medical reference or diagnosis.
DEFAULT_VALUES = {
    "age": 35,
    "urea": 4.5,
    "cr": 60.0,
    "hba1c": 5.2,
    "chol": 4.2,
    "tg": 1.2,
    "hdl": 1.5,
    "ldl": 2.2,
    "vldl": 0.5,
    "bmi": 22.5,
}


# ============================================================
# DATA CLEANING
# ============================================================
def clean_training_data(df):
    data = df.copy()
    data.columns = data.columns.str.strip()

    # Text cleaning
    data["Gender"] = data["Gender"].astype(str).str.strip().str.upper()
    data["CLASS"] = data["CLASS"].astype(str).str.strip().str.upper()

    # Duplicate removal
    data = data.drop_duplicates().reset_index(drop=True)

    # ID columns are not useful model inputs
    data = data.drop(columns=["ID", "No_Pation"], errors="ignore")

    # Keep binary classification rows only
    data = data[data["CLASS"].isin(["N", "Y"])].copy()
    data["CLASS"] = data["CLASS"].map({"N": 0, "Y": 1})

    numeric_cols = [
        "AGE", "Urea", "Cr", "HbA1c",
        "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    # Fix F/f/M style values
    data["Gender"] = data["Gender"].replace({
        "MALE": "M",
        "FEMALE": "F",
    })
    data["Gender"] = data["Gender"].fillna(data["Gender"].mode()[0])

    data = pd.get_dummies(
        data,
        columns=["Gender"],
        drop_first=True,
        dtype=int,
    )

    if "Gender_M" not in data.columns:
        data["Gender_M"] = 0

    data = data[FEATURES + ["CLASS"]]

    return data


# ============================================================
# MODEL
# ============================================================
def build_model():
    model = Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(11, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(6, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


@st.cache_resource(show_spinner=False)
def train_model(csv_bytes):
    raw = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    clean = clean_training_data(raw)

    X = clean[FEATURES].astype(float)
    y = clean["CLASS"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_model()

    model.fit(
        X_train_scaled,
        y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.10,
        verbose=0,
    )

    probability = model.predict(
        X_test_scaled,
        verbose=0
    ).ravel()

    predictions = (probability >= 0.50).astype(int)

    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }

    ranges = {
        col: (
            float(clean[col].min()),
            float(clean[col].max())
        )
        for col in FEATURES
        if col != "Gender_M"
    }

    return model, scaler, clean, scores, ranges


# ============================================================
# DATA
# ============================================================
def find_data():
    for path in DATA_PATHS:
        if path.exists():
            return path
    return None


data_path = find_data()

if data_path is None:
    st.error(
        "Training CSV not found. Keep 'Dataset of Diabetes .csv' beside app.py."
    )
    st.stop()

with st.spinner("Preparing your dashboard..."):
    model, scaler, clean_data, scores, train_ranges = train_model(
        data_path.read_bytes()
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🩺 Diabetes Check")
    st.caption("ANN-powered interactive dashboard")

    st.markdown("---")

    st.markdown("### Quick Guide")
    st.write("1. Start with the example values.")
    st.write("2. Adjust any value from your report.")
    st.write("3. Click **Check Result**.")
    st.write("4. Use the CSV tab for multiple records.")

    st.markdown("---")

    st.markdown("### Data used")
    st.write(f"{len(clean_data):,} cleaned training records")
    st.write("10 health measurements + gender")

    st.markdown("---")
    st.caption(
        "Example values are only a starting point for the interface. "
        "This application is not a medical diagnosis."
    )


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <h1>Diabetes Check</h1>
    <p>Enter values from a lab report and explore how the trained ANN classifies the information.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP CARDS
# ============================================================
a, b, c = st.columns(3)

with a:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Training Data</div>
            <div class="stat-value">{len(clean_data):,} records</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with b:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Input Fields</div>
            <div class="stat-value">11</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Test Accuracy</div>
            <div class="stat-value">{scores['accuracy'] * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TABS
# ============================================================
tab_person, tab_csv, tab_data = st.tabs([
    "👤 Personal Check",
    "📂 Upload CSV",
    "📊 Dataset",
])


# ============================================================
# PERSONAL CHECK
# ============================================================
with tab_person:
    st.markdown(
        '<div class="section-title">Enter Health Details</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="guide">
        <strong>Start with the example profile:</strong> values are already filled in below.
        You can move any slider up or down according to the person's report.
        These example values are only for demonstrating the application and are not a medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Form
    # -----------------------------
    with st.form("personal_prediction"):

        left, right = st.columns(2, gap="large")

        with left:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                index=0,
                help="Select the person's gender."
            )

            age = st.slider(
                "Age",
                min_value=20,
                max_value=79,
                value=DEFAULT_VALUES["age"],
                step=1,
                help="Change the age to match the report."
            )

            urea = st.slider(
                "Urea",
                min_value=0.5,
                max_value=38.9,
                value=DEFAULT_VALUES["urea"],
                step=0.1,
                help="Enter the Urea value from the report."
            )

            cr = st.slider(
                "Creatinine (Cr)",
                min_value=6.0,
                max_value=800.0,
                value=DEFAULT_VALUES["cr"],
                step=1.0,
                help="Enter the Creatinine value from the report."
            )

            hba1c = st.slider(
                "HbA1c",
                min_value=0.9,
                max_value=16.0,
                value=DEFAULT_VALUES["hba1c"],
                step=0.1,
                help="Enter HbA1c from the report."
            )

            chol = st.slider(
                "Cholesterol (Chol)",
                min_value=0.0,
                max_value=10.3,
                value=DEFAULT_VALUES["chol"],
                step=0.1,
                help="Enter Cholesterol from the report."
            )

        with right:
            tg = st.slider(
                "Triglycerides (TG)",
                min_value=0.3,
                max_value=13.8,
                value=DEFAULT_VALUES["tg"],
                step=0.1,
                help="Enter Triglycerides from the report."
            )

            hdl = st.slider(
                "HDL",
                min_value=0.2,
                max_value=9.9,
                value=DEFAULT_VALUES["hdl"],
                step=0.1,
                help="Enter HDL from the report."
            )

            ldl = st.slider(
                "LDL",
                min_value=0.3,
                max_value=9.9,
                value=DEFAULT_VALUES["ldl"],
                step=0.1,
                help="Enter LDL from the report."
            )

            vldl = st.slider(
                "VLDL",
                min_value=0.1,
                max_value=35.0,
                value=DEFAULT_VALUES["vldl"],
                step=0.1,
                help="Enter VLDL from the report."
            )

            bmi = st.slider(
                "BMI",
                min_value=19.0,
                max_value=47.75,
                value=DEFAULT_VALUES["bmi"],
                step=0.1,
                help="Enter BMI from the report."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        check = st.form_submit_button(
            "🔍 Check Result",
            type="primary",
            use_container_width=True
        )

    # -----------------------------
    # Result
    # -----------------------------
    if check:

        input_df = pd.DataFrame([{
            "AGE": age,
            "Urea": urea,
            "Cr": cr,
            "HbA1c": hba1c,
            "Chol": chol,
            "TG": tg,
            "HDL": hdl,
            "LDL": ldl,
            "VLDL": vldl,
            "BMI": bmi,
            "Gender_M": 1 if gender == "Male" else 0,
        }], columns=FEATURES)

        scaled = scaler.transform(input_df)

        score = float(
            model.predict(scaled, verbose=0).ravel()[0]
        )

        diabetic = score >= 0.50

        if diabetic:
            st.markdown("""
            <div class="result result-review">
                <h2>⚠️ Result: Diabetic</h2>
                <p>The model classified the entered information in the diabetic category.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result result-good">
                <h2>✅ Result: Not Diabetic</h2>
                <p>The model classified the entered information in the non-diabetic category.</p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)

        with r1:
            st.metric(
                "Model Output",
                f"{score * 100:.1f}%"
            )

        with r2:
            st.metric(
                "Selected Profile",
                f"{gender}, {age} years"
            )

        st.caption(
            "Model output is for this machine-learning demonstration only "
            "and should not be treated as a medical diagnosis."
        )


# ============================================================
# CSV TAB
# ============================================================
with tab_csv:
    st.markdown(
        '<div class="section-title">Upload Multiple Records</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a CSV containing real-life records and generate predictions in one go.'
        '</div>',
        unsafe_allow_html=True
    )

    template = pd.DataFrame(columns=[
        "AGE", "Urea", "Cr", "HbA1c", "Chol",
        "TG", "HDL", "LDL", "VLDL", "BMI", "Gender"
    ])

    st.download_button(
        "⬇️ Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="diabetes_input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        key="prediction_csv"
    )

    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)

            st.markdown("#### Your Data")
            st.dataframe(
                user_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            def prepare_csv(df):
                data = df.copy()
                data.columns = data.columns.str.strip()

                rename_map = {
                    "Age": "AGE",
                    "age": "AGE",
                    "UREA": "Urea",
                    "urea": "Urea",
                    "Creatinine": "Cr",
                    "creatinine": "Cr",
                    "HBA1C": "HbA1c",
                    "Cholesterol": "Chol",
                    "cholesterol": "Chol",
                    "Triglycerides": "TG",
                    "triglycerides": "TG",
                    "Hdl": "HDL",
                    "hdl": "HDL",
                    "Ldl": "LDL",
                    "ldl": "LDL",
                    "Vldl": "VLDL",
                    "vldl": "VLDL",
                    "Bmi": "BMI",
                    "bmi": "BMI",
                }

                data = data.rename(columns={
                    k: v for k, v in rename_map.items()
                    if k in data.columns
                })

                if "Gender" not in data.columns:
                    raise ValueError(
                        "CSV must contain a Gender column."
                    )

                data["Gender"] = (
                    data["Gender"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace({"MALE": "M", "FEMALE": "F"})
                )

                data = pd.get_dummies(
                    data,
                    columns=["Gender"],
                    drop_first=True,
                    dtype=int
                )

                if "Gender_M" not in data.columns:
                    data["Gender_M"] = 0

                data = data.drop(
                    columns=["ID", "No_Pation", "CLASS"],
                    errors="ignore"
                )

                missing = [
                    col for col in FEATURES
                    if col not in data.columns
                ]

                if missing:
                    raise ValueError(
                        "Missing columns: " + ", ".join(missing)
                    )

                data = data[FEATURES]

                for col in FEATURES:
                    data[col] = pd.to_numeric(
                        data[col], errors="coerce"
                    )

                medians = clean_data[FEATURES].median(
                    numeric_only=True
                )

                data = data.fillna(medians)

                return data

            if st.button(
                "🚀 Generate Predictions",
                type="primary",
                use_container_width=True
            ):
                prepared = prepare_csv(user_df)

                scaled = scaler.transform(prepared)

                probabilities = model.predict(
                    scaled,
                    verbose=0
                ).ravel()

                result = user_df.copy()

                result["Prediction"] = np.where(
                    probabilities >= 0.50,
                    "Diabetic",
                    "Not Diabetic"
                )

                result["Model Output"] = np.round(
                    probabilities * 100, 1
                )

                st.markdown("#### Results")

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Results",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="diabetes_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Could not process the file: {error}")


# ============================================================
# DATASET TAB
# ============================================================
with tab_data:
    st.markdown(
        '<div class="section-title">Training Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A quick view of the cleaned data used to train the ANN.'
        '</div>',
        unsafe_allow_html=True
    )

    preview = clean_data.copy()
    preview["CLASS"] = preview["CLASS"].map({
        0: "Non-Diabetic",
        1: "Diabetic"
    })

    st.dataframe(
        preview.head(15),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Model Performance")

    x1, x2, x3, x4 = st.columns(4)

    with x1:
        st.metric("Accuracy", f"{scores['accuracy'] * 100:.2f}%")
    with x2:
        st.metric("Precision", f"{scores['precision'] * 100:.2f}%")
    with x3:
        st.metric("Recall", f"{scores['recall'] * 100:.2f}%")
    with x4:
        st.metric("F1 Score", f"{scores['f1'] * 100:.2f}%")


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer">'
    'Diabetes Check • Streamlit + TensorFlow/Keras'
    '</div>',
    unsafe_allow_html=True
)


# streamlit run app.py