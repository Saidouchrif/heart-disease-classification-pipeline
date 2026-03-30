import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon=":heart:",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parents[1]
model = joblib.load(BASE_DIR / "models" / "heart_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

DEFAULT_FEATURES = [
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
    "age_group",
]
FEATURE_COLUMNS = list(getattr(scaler, "feature_names_in_", DEFAULT_FEATURES))

st.markdown(
    """
    <style>
    .main-card {
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        padding: 1rem 1rem 0.25rem 1rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    }
    .subtitle {
        color: #4b5563;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Prediction de Maladie Cardiaque")
st.caption("Entrer les informations du patient pour estimer le risque.")

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    with st.form("prediction_form"):
        sex_options = {
            "Femme": 0,
            "Homme": 1,
        }
        chest_pain_options = {
            "Angine typique": 1,
            "Angine atypique": 2,
            "Douleur non angineuse": 3,
            "Asymptomatique": 4,
        }
        fbs_options = {
            "Non": 0,
            "Oui": 1,
        }
        ekg_options = {
            "Normal": 0,
            "Anomalie ST-T": 1,
            "Hypertrophie ventriculaire gauche": 2,
        }
        exercise_angina_options = {
            "Non": 0,
            "Oui": 1,
        }
        slope_options = {
            "Montante": 1,
            "Plate": 2,
            "Descendante": 3,
        }
        thallium_options = {
            "Normal": 3,
            "Defaut fixe": 6,
            "Defaut reversible": 7,
        }

        row1_col1, row1_col2 = st.columns(2)
        age = row1_col1.number_input("Age (ans)", min_value=1, max_value=120, value=50)
        sex_label = row1_col2.selectbox("Sexe", list(sex_options.keys()))
        sex = sex_options[sex_label]

        row2_col1, row2_col2 = st.columns(2)
        chest_pain_label = row2_col1.selectbox("Type de douleur thoracique", list(chest_pain_options.keys()))
        chest_pain = chest_pain_options[chest_pain_label]
        bp = row2_col2.number_input("Pression arterielle (BP)", min_value=0, value=120)

        row3_col1, row3_col2 = st.columns(2)
        cholesterol = row3_col1.number_input("Cholesterol", min_value=0, value=200)
        fbs_label = row3_col2.selectbox("Glycemie a jeun > 120", list(fbs_options.keys()))
        fbs = fbs_options[fbs_label]

        row4_col1, row4_col2 = st.columns(2)
        ekg_label = row4_col1.selectbox("Resultat ECG", list(ekg_options.keys()))
        ekg = ekg_options[ekg_label]
        max_hr = row4_col2.number_input("Frequence cardiaque max (Max HR)", min_value=0, value=150)

        row5_col1, row5_col2 = st.columns(2)
        exercise_angina_label = row5_col1.selectbox("Angine induite par effort", list(exercise_angina_options.keys()))
        exercise_angina = exercise_angina_options[exercise_angina_label]
        st_depression = row5_col2.number_input("Depression du segment ST", min_value=0.0, value=1.0, step=0.1)

        row6_col1, row6_col2 = st.columns(2)
        slope_st_label = row6_col1.selectbox("Pente du segment ST", list(slope_options.keys()))
        slope_st = slope_options[slope_st_label]
        num_vessels = row6_col2.selectbox("Nombre de vaisseaux (fluro)", [0, 1, 2, 3])

        row7_col1, row7_col2 = st.columns(2)
        thallium_label = row7_col1.selectbox("Test Thallium", list(thallium_options.keys()))
        thallium = thallium_options[thallium_label]
        predict_btn = row7_col2.form_submit_button("Predire", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    age_group = age // 10
    input_dict = {
        "Age": age,
        "Sex": sex,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope_st,
        "Number of vessels fluro": num_vessels,
        "Thallium": thallium,
        "age_group": age_group,
    }
    input_data = pd.DataFrame([input_dict]).reindex(columns=FEATURE_COLUMNS)

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Presence de maladie cardiaque ({probability:.2%})")
    else:
        st.success(f"Prediction: Absence de maladie cardiaque ({1 - probability:.2%})")
