import streamlit as st
import numpy as np
import pandas as pd


def main():

    #st.markdown(custom_css, unsafe_allow_html=True)

    # User input fields
    st.title("Loan Default Risk Evaluator")

    # User Input Fields with Default Values

    amt_goods_price = st.number_input("AMT_GOODS_PRICE", value=10, format="%d")
    #credit_term = st.number_input("CREDIT_TERM", value=100, format="%d")
    #days_employed_percent = st.number_input("DAYS_EMPLOYED_PERCENT", value=0, format="%d")
    amt_annuity = st.number_input("AMT_ANNUITY", value=100, format="%d")
    amt_credit = st.number_input("AMT_CREDIT", value=100000, format="%d")
    days_employed = st.number_input("DAYS_EMPLOYED", value=10, format="%d")
    days_birth = st.number_input("DAYS_BIRTH", value=10000, format="%d")
    ext_source_3 = st.number_input("EXT_SOURCE_3", value=0.1, format="%f")
    ext_source_2 = st.number_input("EXT_SOURCE_2", value=0.2, format="%f")
    ext_source_1 = st.number_input("EXT_SOURCE_1", value=0.3, format="%f")

    
    credit_term = amt_annuity / amt_credit

    days_employed_percent = days_employed / days_birth 

    data = {
    'EXT_SOURCE_3': [ext_source_3],
    'EXT_SOURCE_2': [ext_source_2],
    'EXT_SOURCE_1': [ext_source_1],
    'AMT_GOODS_PRICE': [amt_goods_price],
    'CREDIT_TERM': [credit_term],
    'DAYS_EMPLOYED_PERCENT': [days_employed_percent]
    }

    test = pd.DataFrame(data)

    # import pickle
    # with open('model.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    import xgboost
    # Load the saved model
    loaded_model = xgboost.Booster(model_file='xgboost_model.json')

    dtest = xgboost.DMatrix(test, label=test)
    # Make predictions with the loaded model
    predictions = loaded_model.predict(dtest)

    # Make predictions using the loaded model
    #predictions = loaded_model.predict(test)

    print(predictions)
    # Submit button
    if st.button("Submit"):
        # Placeholder condition for demonstration purposes
        if predictions == 0:
            result = "Low Default Risk"
            
        else:
            result = "High Default Risk"
            

        # Display result
        st.write(f"Credit Risk Assessment Result: {result}", unsafe_allow_html=True, key="result")

if __name__ == "__main__":
    main()
