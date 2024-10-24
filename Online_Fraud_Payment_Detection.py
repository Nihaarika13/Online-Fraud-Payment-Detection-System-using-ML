from PIL import Image
from streamlit_option_menu import option_menu
import streamlit as st
import pickle, joblib
import pandas as pd
import base64
# load the pre-trained Random Forest Classifier model
model = joblib.load("fraud_detection_model.joblib")
# load the pre-fitted StandardScaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def header_with_logo():
    # Load logo image
    logo = Image.open('download.jpg')
    
    # Replace with your actual path
    header_html = """
    <style>
    .title-header {
        background-color: #3498db; /* Set your desired background color */
        padding: 100px; /* Adjust padding as needed */
        color: white; /* Set text color */
        border-radius: 5px; /* Optional: Add border radius */
    }
    </style>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    # Display logo and title horizontally
    col1, col2 = st.columns([1, 4])  # Adjust column widths as needed
    with col1:
        st.image(logo, width=125)  # Adjust width of the logo as needed
    with col2:
        st.title("Fraud Payment Detection")  # Replace with your common title

    selected = option_menu(
        menu_title=None,  # required
        options=["Home", "Single Set", "Batch File","Prediction History"],  # required
        icons=["house", "file-alt", "folder-open","clock"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"justify-content": "center", "width": "100%"},
            "nav-link": {"font-size": "16px"},
        }
    )
    if selected == "Home":
        home()
    elif selected == "Single Set":
        predict_one()
    elif selected == "Batch File":
        batch_predict()
    elif selected =="Prediction History":
        predict_history()

def main():
    if "oldBalanceOrg" not in st.session_state:
        st.session_state.oldBalanceOrg = 0.0
    if "newBalanceOrig" not in st.session_state:
        st.session_state.newBalanceOrig = 0.0
    if "amount" not in st.session_state:
        st.session_state.amount = 0.0
    # Initialize session state variables
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'reported_transactions' not in st.session_state:
        st.session_state.reported_transactions = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if "prediction_message" not in st.session_state:
        st.session_state.prediction_message = ""
    if "type" not in st.session_state:
        st.session_state.type = "Select"
    header_with_logo()

# categorical to numerical mapping
type_dict = {"PAYMENT": 0, "TRANSFER": 1, "CASH OUT": 2, "DEBIT": 3, "CASH IN": 4}

def home():
    st.markdown("<h3>WELCOME TO ONLINE FRAUD PAYMENT DETECTION APP</h3>",unsafe_allow_html=True)  
    st.write("A brief information about the project.")
    st.markdown("<h5>Detecting Fraudulent Transactions with Machine Learning</h5>", 
                unsafe_allow_html=True)
    st.markdown(""""
                ## Features
    - Predict the likelihood of fraudulent transactions based on various input parameters.
    - Report suspected fraudulent transactions.
    - Download the list of reported fraudulent transactions as a CSV file.

    ## Uses
    This application is designed to help financial institutions and payment processors detect and report fraudulent transactions in real-time. By leveraging machine learning models, it can identify potentially fraudulent activities based on transaction details provided by the user.

    ## About Model 
    This app utilizes the power of Random Forest Classifer machine learning model to detect fraudulent activity in online transactions.
    This model  has been trained on a comprehensive dataset and has demonstrated a high level of accuracy in detecting fraudulent transactions. Regular updates and retraining of the model ensure that it stays effective in identifying new patterns of fraud.
    
    Accuracy: 0.9999084416773485
    Classification report for model:
                          precision  recall  f1-score   support

                    0       0.99      0.99      0.99     2349
                    1       0.99      0.99      0.99     2333

           accuracy                             0.99     4682
             macro avg       0.99      0.99     0.99     4682
          weighted avg       0.99      0.99     0.99     4682            
    
        """)
def predict_transaction(type, amount, oldBalOrg, newBalOrg):
    data = [
        [type_dict[type], amount, oldBalOrg, newBalOrg]
    ]
    columns = [
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
    ]
    df = pd.DataFrame(data, columns=columns)
    df[columns] = scaler.transform(df)
    prediction = model.predict(df)
    return prediction

def predict_one():
    if "oldBalanceOrg" not in st.session_state:
        st.session_state.oldBalanceOrg = 0.0
    if "newBalanceOrig" not in st.session_state:
        st.session_state.newBalanceOrig = 0.0
    if "amount" not in st.session_state:
        st.session_state.amount = 0.0
    if "prediction_message" not in st.session_state:
        st.session_state.prediction_message = ""
    # this function returns the prediction on a single transaction
    st.write("Here you can check if a transaction is fraud or not.")
    st.header("📈Single Set Prediction")
    
    if st.button("Clear inputs"):
        st.session_state.update({"oldBalanceOrg": 0.0, "newBalanceOrig": 0.0, "amount": 0.0,"type": "Select", "prediction_message": ""})
        st.experimental_rerun()

    type = st.selectbox('Transaction Type', ["Select","PAYMENT", "TRANSFER", "CASH OUT", "DEBIT", "CASH IN"], key='type')
    amount = st.number_input(
        "Amount", min_value=0.0, key="amount", value=st.session_state.amount
    )
    oldBalanceOrg = st.number_input(
            "Old balance at origin",
            min_value=0.0,
            key="oldBalanceOrg",
            value=st.session_state.oldBalanceOrg,
    )
    newBalanceOrig = st.number_input(
            "New balance at origin",
            min_value=0.0,
            key="newBalanceOrig",
            value=st.session_state.newBalanceOrig,
    )

    if st.button("Predict"):
        if type=="Select" and amount == 0.0 and oldBalanceOrg == 0.0 and newBalanceOrig == 0.0:
            st.warning("Please fill all the required fields.")
        else:
            isFraud = predict_transaction(type, amount, oldBalanceOrg, newBalanceOrig)
            # Store result in session state
            st.session_state.prediction_result = isFraud
            st.session_state.prediction_history.append({
                "Transaction Type": type,
                "Amount": amount,
                "Old Balance Origin": oldBalanceOrg,
                "New Balance Origin": newBalanceOrig,
                "Prediction": "Fraudulent" if isFraud == 1 else "Non-Fraudulent",
                "Transaction ID": "",
                "Account ID": ""
            })

        # Display prediction result and report section
    if st.session_state.prediction_result is not None:
        if st.session_state.prediction_result == 1:
            st.error("Prediction: " + "⚠️ The transaction is predicted to be fraudulent.")
            report_expander = st.expander("Report Fraudulent Transaction")
            with report_expander:
                transaction_id = st.text_input("Enter Transaction ID", "")
                account_id = st.text_input("Enter Account ID", "")
                if transaction_id and account_id:
                    confirm_report = st.button("Confirm Report")
                    if confirm_report:
                        st.session_state.reported_transactions.append({
                            "Transaction ID": transaction_id,
                            "Account ID": account_id,
                            "Transaction Type": type,
                            "Amount": amount,
                            "Old Balance Origin": oldBalanceOrg,
                            "New Balance Origin": newBalanceOrig,
                            "Prediction": "Fraudulent"
                        })
                        st.success(f"Transaction {transaction_id} reported.")
                        # Update the prediction history with the reported transaction details
                        st.session_state.prediction_history[-1]["Transaction ID"] = transaction_id
                        st.session_state.prediction_history[-1]["Account ID"] = account_id
        else:
            st.success("Prediction: " + "✅ The transaction is predicted to be non-fraudulent.")

def batch_predict():
    # valid attributes for the model and scaler
    columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']
    # mapping prediction
    output = {0: 'Not Fraud', 1: 'Fraud'}

    # this function runs predicition on every record in the dataset
    def predict_batch(data):
        return model.predict(data)

    # streamlit UI
    header = st.container()

    with header:
        st.title("📊Batch File Prediction")
        st.write("Here you can check multiple transactions from a single CSV file.")

    with st.expander("Upload a .csv file to predict transactions", expanded=True):
        uploaded_file = st.file_uploader("Choose file:", type="csv")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            st.write(f"There are {df.shape[0]} records and {df.shape[1]} attributes.")
            if set(df.columns) != set(columns):
                st.error(f"File columns do not match the expected columns. Expected columns: {columns}.")

        if st.button("Predict file"):
            if uploaded_file:
                new_df = df[columns]
                new_df['type'] = df['type'].replace(type_dict)
                new_df[columns] = scaler.transform(new_df)
                predictions = predict_batch(new_df)
                df['isFraud'] = predictions
                df['isFraud'] = df['isFraud'].replace(output)
                counts = df['isFraud'].value_counts()
                st.success(f"File has {counts[0]} legitimate transactions ✅ and {counts[1]} fraud transactions ⚠️.")
                st.download_button(label="Download as CSV", data=df.to_csv(index=False), file_name='predicted_transactions.csv', mime='text/csv')
            else:
                st.warning("Please upload the file")
def predict_history():
    st.title("Prediction History")

    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)

        # Download button
        csv = history_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("No predictions made yet.")


if __name__ == "__main__":
    main()
