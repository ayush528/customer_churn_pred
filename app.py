import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load models at startup
@st.cache_resource
def load_models():
    model = joblib.load('models/churn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)
    return model, scaler, feature_names, preprocessing_info

model, scaler, feature_names, preprocessing_info = load_models()

# ADD THESE TWO FUNCTIONS FROM YOUR main.py:

def create_engineered_features(df):
    """Create all 8 engineered features"""
    # Initialize columns
    df['SENIOR/YOUNG_GENDER'] = None
    df['PHONE_SER_GENDER'] = None
    df['DEPEND_GENDER'] = None
    df['PHONE_LINE_GENDER'] = None
    df['GENDER_PAYMENT'] = None
    df['GENDER_CONTRACT'] = None
    df['SEN_YNG_GENDER_INTERNET_SER'] = None
    df['INT_SEC_SERV_GENDER'] = None
    
    # 1. SENIOR/YOUNG_GENDER
    df.loc[(df['gender'] == 0) & (df['SeniorCitizen'] == 1), 'SENIOR/YOUNG_GENDER'] = 'senior_male'
    df.loc[(df['gender'] == 0) & (df['SeniorCitizen'] == 0), 'SENIOR/YOUNG_GENDER'] = 'young_male'
    df.loc[(df['gender'] == 1) & (df['SeniorCitizen'] == 1), 'SENIOR/YOUNG_GENDER'] = 'senior_female'
    df.loc[(df['gender'] == 1) & (df['SeniorCitizen'] == 0), 'SENIOR/YOUNG_GENDER'] = 'young_female'
    
    # 2. PHONE_SER_GENDER
    df.loc[(df['gender'] == 0) & (df['PhoneService'] == 'Yes'), 'PHONE_SER_GENDER'] = 'phone_ser_male'
    df.loc[(df['gender'] == 0) & (df['PhoneService'] == 'No'), 'PHONE_SER_GENDER'] = 'no_phone_ser_male'
    df.loc[(df['gender'] == 1) & (df['PhoneService'] == 'Yes'), 'PHONE_SER_GENDER'] = 'phone_service_female'
    df.loc[(df['gender'] == 1) & (df['PhoneService'] == 'No'), 'PHONE_SER_GENDER'] = 'no_phone_ser_female'
    
    # 3. DEPEND_GENDER
    df.loc[(df['gender'] == 0) & (df['Dependents'] == 'Yes'), 'DEPEND_GENDER'] = 'dependent_male'
    df.loc[(df['gender'] == 0) & (df['Dependents'] == 'No'), 'DEPEND_GENDER'] = 'undependent_male'
    df.loc[(df['gender'] == 1) & (df['Dependents'] == 'Yes'), 'DEPEND_GENDER'] = 'dependent_female'
    df.loc[(df['gender'] == 1) & (df['Dependents'] == 'No'), 'DEPEND_GENDER'] = 'undependent_female'
    
    # 4. PHONE_LINE_GENDER
    df.loc[(df['gender'] == 0) & (df['MultipleLines'] == 'Yes'), 'PHONE_LINE_GENDER'] = 'multiple_lines_male'
    df.loc[(df['gender'] == 0) & (df['MultipleLines'] == 'No'), 'PHONE_LINE_GENDER'] = 'single_line_male'
    df.loc[(df['gender'] == 0) & (df['MultipleLines'] == 'No phone service'), 'PHONE_LINE_GENDER'] = 'no_line_male'
    df.loc[(df['gender'] == 1) & (df['MultipleLines'] == 'Yes'), 'PHONE_LINE_GENDER'] = 'multiple_lines_female'
    df.loc[(df['gender'] == 1) & (df['MultipleLines'] == 'No'), 'PHONE_LINE_GENDER'] = 'single_line_female'
    df.loc[(df['gender'] == 1) & (df['MultipleLines'] == 'No phone service'), 'PHONE_LINE_GENDER'] = 'no_line_female'
    
    # 5. GENDER_PAYMENT
    df.loc[(df['gender'] == 0) & (df['PaymentMethod'] == 'Electronic check'), 'GENDER_PAYMENT'] = 'male_electronic_check_pay'
    df.loc[(df['gender'] == 0) & (df['PaymentMethod'] == 'Mailed check'), 'GENDER_PAYMENT'] = 'male_mailed_check_pay'
    df.loc[(df['gender'] == 0) & (df['PaymentMethod'] == 'Bank transfer (automatic)'), 'GENDER_PAYMENT'] = 'male_bank_transfer_pay'
    df.loc[(df['gender'] == 0) & (df['PaymentMethod'] == 'Credit card (automatic)'), 'GENDER_PAYMENT'] = 'male_credit_card_pay'
    df.loc[(df['gender'] == 1) & (df['PaymentMethod'] == 'Electronic check'), 'GENDER_PAYMENT'] = 'female_electronic_check_pay'
    df.loc[(df['gender'] == 1) & (df['PaymentMethod'] == 'Mailed check'), 'GENDER_PAYMENT'] = 'female_mailed_check_pay'
    df.loc[(df['gender'] == 1) & (df['PaymentMethod'] == 'Bank transfer (automatic)'), 'GENDER_PAYMENT'] = 'female_bank_transfer_pay'
    df.loc[(df['gender'] == 1) & (df['PaymentMethod'] == 'Credit card (automatic)'), 'GENDER_PAYMENT'] = 'female_credit_card_pay'
    
    # 6. GENDER_CONTRACT
    df.loc[(df['gender'] == 0) & (df['Contract'] == 'Month-to-month'), 'GENDER_CONTRACT'] = 'male_monthly_contract'
    df.loc[(df['gender'] == 0) & (df['Contract'] == 'One year'), 'GENDER_CONTRACT'] = 'male_one_year_contact'
    df.loc[(df['gender'] == 0) & (df['Contract'] == 'Two year'), 'GENDER_CONTRACT'] = 'male_two_year_contact'
    df.loc[(df['gender'] == 1) & (df['Contract'] == 'Month-to-month'), 'GENDER_CONTRACT'] = 'female_monthly_contract'
    df.loc[(df['gender'] == 1) & (df['Contract'] == 'One year'), 'GENDER_CONTRACT'] = 'female_one_year_contact'
    df.loc[(df['gender'] == 1) & (df['Contract'] == 'Two year'), 'GENDER_CONTRACT'] = 'female_two_year_contact'
    
    # 7. SEN_YNG_GENDER_INTERNET_SER
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_male') & (df['InternetService'] == 'Fiber optic'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_male_fiber_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_male') & (df['InternetService'] == 'DSL'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_male_dsl_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_male') & (df['InternetService'] == 'No'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_male_no_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_male') & (df['InternetService'] == 'Fiber optic'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_male_fiber_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_male') & (df['InternetService'] == 'DSL'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_male_dsl_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_male') & (df['InternetService'] == 'No'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_male_no_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_female') & (df['InternetService'] == 'Fiber optic'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_female_fiber_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_female') & (df['InternetService'] == 'DSL'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_female_dsl_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'senior_female') & (df['InternetService'] == 'No'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'senior_female_no_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_female') & (df['InternetService'] == 'Fiber optic'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_female_fiber_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_female') & (df['InternetService'] == 'DSL'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_female_dsl_internet'
    df.loc[(df['SENIOR/YOUNG_GENDER'] == 'young_female') & (df['InternetService'] == 'No'), 'SEN_YNG_GENDER_INTERNET_SER'] = 'young_female_no_internet'
    
    # 8. INT_SEC_SERV_GENDER
    df.loc[(df['gender'] == 0) & (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'Yes'), 'INT_SEC_SERV_GENDER'] = 'male_fiber_int_security'
    df.loc[(df['gender'] == 0) & (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No'), 'INT_SEC_SERV_GENDER'] = 'male_fiber_int_no_security'
    df.loc[(df['gender'] == 0) & (df['InternetService'] == 'DSL') & (df['OnlineSecurity'] == 'Yes'), 'INT_SEC_SERV_GENDER'] = 'male_dsl_int_security'
    df.loc[(df['gender'] == 0) & (df['InternetService'] == 'DSL') & (df['OnlineSecurity'] == 'No'), 'INT_SEC_SERV_GENDER'] = 'male_dsl_int_no_security'
    df.loc[(df['gender'] == 1) & (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'Yes'), 'INT_SEC_SERV_GENDER'] = 'female_fiber_int_security'
    df.loc[(df['gender'] == 1) & (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No'), 'INT_SEC_SERV_GENDER'] = 'female_fiber_int_no_security'
    df.loc[(df['gender'] == 1) & (df['InternetService'] == 'DSL') & (df['OnlineSecurity'] == 'Yes'), 'INT_SEC_SERV_GENDER'] = 'female_dsl_int_security'
    df.loc[(df['gender'] == 1) & (df['InternetService'] == 'DSL') & (df['OnlineSecurity'] == 'No'), 'INT_SEC_SERV_GENDER'] = 'female_dsl_int_no_security'

def preprocess_customer(customer_data):
    """Preprocess exactly as training script does"""
    df = pd.DataFrame([customer_data])
    
    # Handle TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        if df['TotalCharges'].isna().any():
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    
    # Step 1: Encode gender FIRST
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    
    # Step 2: Create engineered features
    create_engineered_features(df)
    
    # Step 3: Label encode binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Step 4: One-hot encode remaining categorical columns
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaymentMethod', 'SENIOR/YOUNG_GENDER', 'PHONE_SER_GENDER',
                'DEPEND_GENDER', 'PHONE_LINE_GENDER', 'GENDER_PAYMENT', 'GENDER_CONTRACT',
                'SEN_YNG_GENDER_INTERNET_SER', 'INT_SEC_SERV_GENDER']
    existing_ohe_cols = [col for col in ohe_cols if col in df.columns and df[col].dtype == 'object']
    if existing_ohe_cols:
        df = pd.get_dummies(df, columns=existing_ohe_cols, drop_first=True)
    
    # Step 5: Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if all(col in df.columns for col in num_cols):
        df[num_cols] = scaler.transform(df[num_cols])
    
    # Step 6: Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Step 7: Align with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    return df
# Load models at startup
@st.cache_resource
def load_models():
    model = joblib.load('models/churn_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('models/preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)
    return model, scaler, feature_names, preprocessing_info

model, scaler, feature_names, preprocessing_info = load_models()

# Page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")
st.title("📊 Customer Churn Prediction")

# Copy your preprocess_customer and create_engineered_features functions here
# (from your main.py)

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    
    with col2:
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    
    total_charges = st.number_input("Total Charges", min_value=0.0, value=monthly_charges * tenure)
    
    submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        customer_data = {
            'customerID': 'CUST001',
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Preprocess and predict
        processed_data = preprocess_customer(customer_data)
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        churn_prob = probabilities[1]
        
        # Display results
        st.subheader("Prediction Results")
        
        if churn_prob > 0.7:
            st.error(f"⚠️ HIGH RISK: {churn_prob*100:.1f}% chance of churn")
            st.write("**Recommendation:** Immediate retention with 25% discount + free tech support")
        elif churn_prob > 0.4:
            st.warning(f"⚠️ MEDIUM RISK: {churn_prob*100:.1f}% chance of churn")
            st.write("**Recommendation:** Monitor closely with proactive check-in within 48 hours")
        else:
            st.success(f"✅ LOW RISK: {churn_prob*100:.1f}% chance of churn")
            st.write("**Recommendation:** Maintain regular touchpoints and satisfaction surveys")

