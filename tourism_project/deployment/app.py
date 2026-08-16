import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(layout="wide")

# Load the trained model
# Model is saved in the same directory as app.py within the deployment folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model not found at {MODEL_PATH}. Please ensure the model is trained and saved in the correct deployment directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Wellness Tourism Package Prediction")
st.markdown("### Predict if a customer will purchase the Wellness Tourism Package")

st.sidebar.header("Customer Input Features")

# Collect user input features
def user_input_features():
    age = st.sidebar.slider("Age", 18, 90, 35)
    typeofcontact = st.sidebar.selectbox("Type of Contact", ('Self Enquiry', 'Company Invited'))
    citytier = st.sidebar.selectbox("City Tier", (1, 2, 3))
    durationofpitch = st.sidebar.slider("Duration of Pitch (minutes)", 0, 60, 10)
    occupation = st.sidebar.selectbox("Occupation", ('Salaried', 'Small Business', 'Large Business', 'Free Lancer'))
    gender = st.sidebar.selectbox("Gender", ('Male', 'Female', 'Fe Male'))
    numberofpersonvisiting = st.sidebar.slider("Number of Persons Visiting", 1, 10, 2)
    numberoffollowups = st.sidebar.slider("Number of Follow-ups", 0, 10, 3)
    productpitched = st.sidebar.selectbox("Product Pitched", ('Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'))
    preferredpropertystar = st.sidebar.selectbox("Preferred Property Star", (3, 4, 5))
    maritalstatus = st.sidebar.selectbox("Marital Status", ('Single', 'Married', 'Divorced'))
    numberoftrips = st.sidebar.slider("Number of Trips (annually)", 1, 20, 3)
    passport = st.sidebar.selectbox("Passport", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    pitchsatisfactionsccore = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
    owncar = st.sidebar.selectbox("Own Car", (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    numberofchildrenvisiting = st.sidebar.slider("Number of Children Visiting", 0, 5, 0)
    designation = st.sidebar.selectbox("Designation", ('Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'))
    monthlyincome = st.sidebar.number_input("Monthly Income", min_value=0.0, value=25000.0, step=1000.0)

    data = {
        'Age': age,
        'TypeofContact': typeofcontact,
        'CityTier': citytier,
        'DurationOfPitch': durationofpitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': numberofpersonvisiting,
        'NumberOfFollowups': numberoffollowups,
        'ProductPitched': productpitched,
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': maritalstatus,
        'NumberOfTrips': numberoftrips,
        'Passport': passport,
        'PitchSatisfactionScore': pitchsatisfactionsccore,
        'OwnCar': owncar,
        'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': designation,
        'MonthlyIncome': monthlyincome,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    if prediction[0] == 1:
        st.success("The customer is likely to purchase the Wellness Tourism Package!")
    else:
        st.info("The customer is unlikely to purchase the Wellness Tourism Package.")

    st.subheader('Prediction Probability')
    prediction_df = pd.DataFrame({
        'Class': ['Not Purchased', 'Purchased'],
        'Probability': prediction_proba[0]
    })
    st.write(prediction_df)

    st.bar_chart(prediction_df.set_index('Class'))
