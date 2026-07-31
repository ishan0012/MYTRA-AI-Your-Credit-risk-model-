import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="MYTRA-AI, A Credit Default Prediction AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM FINTECH CSS
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Reset & Typography */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Hide default Streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography Overrides */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
    
    /* Container Cards */
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Custom KPI Cards */
    .kpi-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .kpi-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    
    /* Action Button */
    div.stButton > button:first-child {
        background: #0f172a;
        color: #ffffff;
        font-weight: 600;
        font-size: 1.05rem;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        height: 3.5rem;
        box-shadow: 0 4px 10px 0 rgba(15, 23, 42, 0.2);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: #1e293b;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px 0 rgba(15, 23, 42, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA MAPPINGS
# -----------------------------
status_map = {"No Checking Account": 0, "< 0 DM": 1, "0 to 200 DM": 2, ">= 200 DM (Salary Account)": 3}
history_map = {"No credits taken / All paid back": 0, "All credits at this bank paid back duly": 1, "Existing credits paid back duly till now": 2, "Delay in paying off in the past": 3, "Critical account / Other credits existing": 4}
purpose_map = {"Car (New)": 0, "Car (Used)": 1, "Furniture/Equipment": 2, "Radio/Television": 3, "Domestic Appliances": 4, "Repairs": 5, "Education": 6, "Vacation": 7, "Retraining": 8, "Business": 9, "Others": 10}
savings_map = {"Unknown / No savings account": 0, "< 100 DM": 1, "100 to 500 DM": 2, "500 to 1000 DM": 3, ">= 1000 DM": 4}
emp_map = {"Unemployed": 0, "< 1 year": 1, "1 to 4 years": 2, "4 to 7 years": 3, ">= 7 years": 4}
sex_map = {"Male: Divorced/Separated": 0, "Female: Divorced/Separated/Married": 1, "Male: Single": 2, "Male: Married/Widowed": 3}
debtor_map = {"None": 0, "Co-applicant": 1, "Guarantor": 2}
property_map = {"Real Estate": 0, "Life Insurance / Building Society": 1, "Car / Other": 2, "Unknown / No Property": 3}
install_map = {"Bank": 0, "Stores": 1, "None": 2}
housing_map = {"Rent": 0, "Own": 1, "For Free": 2}
job_map = {"Unemployed / Unskilled (Non-resident)": 0, "Unskilled (Resident)": 1, "Skilled Employee": 2, "Management / Self-Employed": 3}
foreign_map = {"Yes (Foreign Worker)": 0, "No": 1}

# -----------------------------
# MODEL LOADING
# -----------------------------
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load("models/credit_risk_model.pkl")
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except Exception as e:
        return None, None

model, explainer = load_ml_assets()
model_loaded = model is not None

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ CreditAI Settings")
    st.markdown("---")
    
    risk_threshold = st.slider(
        "Default Risk Tolerance (%)", 
        min_value=10.0, max_value=90.0, value=50.0, step=5.0,
        help="Adjust the threshold that triggers a High-Risk Flag."
    )
    
    st.markdown("---")
    with st.expander("System Diagnostics", expanded=True):
        st.markdown("**Algorithm:** CatBoost Classifier")
        st.markdown("**Features:** 23 Input Variables")
        st.markdown("**Training Data:** German Credit")
        st.markdown("**Interpretability:** SHAP TreeExplainer")
        
        if model_loaded:
            st.success("🟢 API Status: Online")
        else:
            st.error("🔴 API Status: Offline")

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 3rem; border-radius: 12px; color: white; margin-bottom: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);">
    <span style="background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; border: 1px solid rgba(59, 130, 246, 0.3);">
        Risk Core v1.2
    </span>
    <h1 style="color: white; margin-top: 1rem; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 800;">MYTRA AI-Your finance Guide</h1>
    <p style="color: #94a3b8; margin: 0; font-size: 1.1rem; max-width: 800px;">
        Institutional risk matrix evaluation powered by a verified CatBoost ensemble framework.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# APPLICANT INPUT FORM
# -----------------------------
st.markdown("### Applicant Profile Setup")
tab_fin, tab_demo, tab_assets = st.tabs(["💳 Financials & Loan", "👤 Demographics & Work", "🏠 Assets & History"])

with tab_fin:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        v_amount = st.number_input("Credit Amount ($)", min_value=250, value=5000, step=250)
        v_duration = st.slider("Loan Term (Months)", 4, 72, 24)
        v_monthly = st.number_input("Est. Monthly Payment ($)", min_value=0.0, value=220.0, step=10.0)
    with c2:
        v_status = st.selectbox("Checking Status", list(status_map.keys()), help="Current standing of applicant's primary account.")
        v_savings = st.selectbox("Savings Balance", list(savings_map.keys()), help="Verified liquid assets.")
    with c3:
        v_purpose = st.selectbox("Capital Purpose", list(purpose_map.keys()))
        v_install_rate = st.slider("Installment Rate (%)", 1, 4, 2, help="Percentage of disposable income allocated to this payment.")
        v_high_cred = st.selectbox("Pre-screen Flag", ["Standard", "High Exposure"])
    st.markdown("</div>", unsafe_allow_html=True)

with tab_demo:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        v_age = st.number_input("Applicant Age", min_value=18, max_value=90, value=35)
        v_age_grp = st.selectbox("Age Group Tier", ["0 (Youth)", "1 (Young Adult)", "2 (Adult)", "3 (Senior)"])
        v_sex = st.selectbox("Civil Status & Gender", list(sex_map.keys()))
    with c5:
        v_emp = st.selectbox("Employment Duration", list(emp_map.keys()))
        v_job = st.selectbox("Job Classification", list(job_map.keys()))
    with c6:
        v_liable = st.slider("Number of Dependents", 1, 2, 1)
        v_foreign = st.selectbox("Foreign Worker Status", list(foreign_map.keys()))
        v_phone = st.selectbox("Contact Verification", ["Unverified/None", "Verified / Registered"])
    st.markdown("</div>", unsafe_allow_html=True)

with tab_assets:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        v_history = st.selectbox("Historical Payment Behavior", list(history_map.keys()))
        v_exist_cred = st.slider("Active Lines at this Bank", 1, 4, 1)
    with c8:
        v_housing = st.selectbox("Housing Status", list(housing_map.keys()))
        v_residence = st.slider("Years at Current Residence", 1, 4, 2)
        v_property = st.selectbox("Primary Asset", list(property_map.keys()))
    with c9:
        v_other_debt = st.selectbox("Other Debtors / Guarantors", list(debtor_map.keys()))
        v_other_inst = st.selectbox("External Installment Plans", list(install_map.keys()))
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
predict = st.button("Execute Credit Assessment Pipeline", use_container_width=True)

# -----------------------------
# PREDICTION PROCESSOR
# -----------------------------
if predict and model_loaded:
    input_data = pd.DataFrame([{
        "Status": status_map[v_status],
        "Duration": v_duration,
        "Credit_History": history_map[v_history],
        "Purpose": purpose_map[v_purpose],
        "Credit_Amount": v_amount,
        "Savings": savings_map[v_savings],
        "Employment": emp_map[v_emp],
        "Installment_Rate": v_install_rate,
        "Personal_Status_Sex": sex_map[v_sex],
        "Other_Debtors": debtor_map[v_other_debt],
        "Residence_Since": v_residence,
        "Property": property_map[v_property],
        "Age": v_age,
        "Other_Installment": install_map[v_other_inst],
        "Housing": housing_map[v_housing],
        "Existing_Credits": v_exist_cred,
        "Job": job_map[v_job],
        "People_Liable": v_liable,
        "Telephone": 1 if v_phone == "Verified / Registered" else 0,
        "Foreign_Worker": foreign_map[v_foreign],
        "Monthly_Loan": v_monthly,
        "Age_Group": int(v_age_grp.split(" ")[0]),
        "High_Credit": 1 if v_high_cred == "High Exposure" else 0
    }])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
    except Exception as e:
        st.error(f"Execution Error: Ensure your model expects exactly 23 features. Details: {e}")
        st.stop()

    bad_prob = probability[0] * 100
    good_prob = probability[1] * 100
    confidence = max(good_prob, bad_prob)

    st.markdown("---")
    
    # 1. Executive Decision Banner
    if bad_prob >= risk_threshold:
        decision_color = "#dc2626"
        decision_bg = "rgba(220, 38, 38, 0.1)"
        decision_text = "DECLINED"
        decision_subtext = f"Calculated default probability ({bad_prob:.2f}%) exceeds the platform tolerance threshold."
    else:
        decision_color = "#059669"
        decision_bg = "rgba(5, 150, 105, 0.1)"
        decision_text = "APPROVED"
        decision_subtext = f"Applicant profile aligns with standard credit risk parameters ({good_prob:.2f}% solvency)."

    st.markdown(f"""
    <div style="background-color: {decision_bg}; border-left: 6px solid {decision_color}; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
        <h2 style="color: {decision_color}; margin: 0; font-weight: 800;">{decision_text}</h2>
        <p style="margin: 0.25rem 0 0 0; font-size: 1.1rem; color: #475569;">{decision_subtext}</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. KPI Metrics
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #10b981;">
            <div class="card-label">Approval Probability</div>
            <div class="card-value" style="color: #10b981;">{good_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ef4444;">
            <div class="card-label">Default Risk Boundary</div>
            <div class="card-value" style="color: #ef4444;">{bad_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #3b82f6;">
            <div class="card-label">Model Confidence Index</div>
            <div class="card-value" style="color: #3b82f6;">{confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # 3. SHAP Explainability
    st.write("")
    st.markdown("### 🧠 AI Decision Explainer")
    
    with st.spinner('Generating real-time SHAP attribution metrics...'):
        shap_values = explainer.shap_values(input_data)
        
        if isinstance(shap_values, list):
            target_shap = shap_values[1]
            base_value = explainer.expected_value[1]
        else:
            target_shap = shap_values
            base_value = explainer.expected_value

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["Local Vector Force Graph", "Global Weight Analysis"])
    
    with t1:
        st.write("Red arrows increase probability of Good Credit; Blue arrows push toward Default.")
        st_shap(shap.force_plot(base_value, target_shap[0, :], input_data.iloc[0, :]), height=150)
        
    with t2:
        st.write("Feature attribution hierarchy for this specific profile.")
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.summary_plot(target_shap, input_data, plot_type="bar", show=False)
        st.pyplot(fig, use_container_width=True)
        plt.clf()
    st.markdown("</div>", unsafe_allow_html=True)

    # 4. Data Export
    with st.expander("📄 Export Application JSON Schema & Logging"):
        st.dataframe(input_data, use_container_width=True)
        st.download_button(
            label="Download Application Payload (.CSV)",
            data=input_data.to_csv(index=False),
            file_name="underwriting_payload.csv",
            mime="text/csv"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#94a3b8; padding:10px; font-size:0.85rem;'>
    Developed by Ishan Arora | Machine Learning Underwriting Engine
</div>
""", unsafe_allow_html=True)