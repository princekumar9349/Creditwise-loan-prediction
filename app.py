

import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CreditWise", page_icon="🏦", layout="wide")

# trained model aur uski details load 
model = joblib.load("model.joblib")
info = json.load(open("meta.json"))

st.title("CreditWise - Loan Approval Predictor")
st.write("Model used:", info["best"])

tab1, tab2, tab3 = st.tabs(["Predict", "Model Performance", "Dataset"])


# ---------- TAB 1 : prediction ----------
with tab1:
    st.header("Enter applicant details")

    col1, col2, col3 = st.columns(3)

    # first column
    applicant_income = col1.number_input("Applicant Income", 2000, 20000, 10000)
    coapplicant_income = col1.number_input("Coapplicant Income", 0, 10000, 5000)
    credit_score = col1.slider("Credit Score", 550, 800, 680)
    dti_ratio = col1.slider("DTI Ratio", 0.1, 0.6, 0.35)

    # second column
    savings = col2.number_input("Savings", 0, 20000, 10000)
    collateral = col2.number_input("Collateral Value", 0, 50000, 25000)
    loan_amount = col2.number_input("Loan Amount", 1000, 40000, 20000)
    loan_term = col2.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)

    # third column
    age = col3.slider("Age", 21, 59, 40)
    dependents = col3.slider("Dependents", 0, 3, 1)
    existing_loans = col3.slider("Existing Loans", 0, 4, 2)
    employment = col3.selectbox("Employment Status", ["Salaried", "Contract", "Self-employed", "Unemployed"])

    # fourth row
    col4, col5, col6 = st.columns(3)
    purpose = col4.selectbox("Loan Purpose", ["Home", "Car", "Education", "Business", "Personal"])
    area = col5.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    education = col6.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employer = col4.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])

    if st.button("Predict"):
        # model ko dataframe chahiye, column names wahi hone chahiye jo training mein the
        data = pd.DataFrame({
            "Applicant_Income": [applicant_income],
            "Coapplicant_Income": [coapplicant_income],
            "Age": [age],
            "Dependents": [dependents],
            "Credit_Score": [credit_score],
            "Existing_Loans": [existing_loans],
            "DTI_Ratio": [dti_ratio],
            "Savings": [savings],
            "Collateral_Value": [collateral],
            "Loan_Amount": [loan_amount],
            "Loan_Term": [loan_term],
            "Employment_Status": [employment],
            "Loan_Purpose": [purpose],
            "Property_Area": [area],
            "Education_Level": [education],
            "Employer_Category": [employer],
        })

        # probability of class 1 (approved)
        prob = model.predict_proba(data)[0][1]

        if prob >= 0.5:
            st.success("Loan will probably be APPROVED")
        else:
            st.error("Loan will probably be REJECTED")

        st.write("Approval probability:", round(prob * 100, 1), "%")
        st.progress(float(prob))


# ---------- TAB 2 : model performance ----------
with tab2:
    st.header("Model results")
    st.write("Results on test data (20% of the dataset)")

    results = pd.DataFrame(info["report"]).T
    results = results.drop(columns="cm")
    st.dataframe(results.round(3))

    st.subheader("Most important features")
    st.bar_chart(pd.Series(info["importance"]))

    st.subheader("Confusion matrix")
    cm = info["report"][info["best"]]["cm"]
    cm_df = pd.DataFrame(cm, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])
    st.dataframe(cm_df)


# ---------- TAB 3 : dataset ----------
with tab3:
    st.header("Dataset")
    df = pd.read_csv("loan_approval_data.csv")
    st.write("Total rows:", len(df))
    st.dataframe(df.head(50))