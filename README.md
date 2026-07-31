💳 MYTRA-AI: Credit Default Prediction Engine
MYTRA-AI is an advanced financial credit risk scoring application designed to evaluate loan applicant solvency in real-time. Powered by a verified CatBoost ensemble framework and utilizing the German Credit dataset, this tool provides institutional-grade risk matrix evaluations alongside transparent AI decision-making.

✨ Key Features
Real-Time Risk Assessment: Instantly calculates the probability of default based on 23 financial and demographic features.
AI Explainability (XAI): Integrates SHAP (SHapley Additive exPlanations) to provide local vector force graphs and global weight analysis, ensuring model transparency.
Interactive Fintech UI: Built with a custom-styled Streamlit interface featuring dynamic KPI cards and visual risk indicators.
Executive Decision Engine: Automatically flags high-risk applications based on customizable risk tolerance thresholds.
Data Export: Generates and downloads application payloads (CSV) for downstream underwriting systems.

🛠️ Technology Stack
Frontend UI: Streamlit, Custom CSS
Machine Learning: CatBoost Classifier
Model Explainability: SHAP, streamlit-shap
Data Processing: Pandas, NumPy
Visualization: Matplotlib

🚀 Local Installation
1. Clone the repositor
   
3. Verify the Model Architecture
   Ensure that your pre-trained model file is located precisely at `models/credit_risk_model.pkl`.
4. Run the Application

👨‍💻 Developer
Ishan Arora

