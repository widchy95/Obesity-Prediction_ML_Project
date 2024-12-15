import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('best_obesity_model.pkl')

# Create an instance of LabelEncoder to encode categorical data
label_encoder = LabelEncoder()

# Define a function for data preprocessing
def preprocess_input(data):
    # Example of encoding gender and other categorical variables
    data['gender'] = label_encoder.fit_transform([data['gender']])[0]
    data['family_history_with_overweight'] = int(data['family_history_with_overweight'])
    data['high_calorie_intake'] = int(data['high_calorie_intake'])
    
    # Process 'food_between_meals' safely without calling .lower() on non-string values
    if isinstance(data['food_between_meals'], str):
        data['food_between_meals'] = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[data['food_between_meals'].lower()]
    else:
        data['food_between_meals'] = int(data['food_between_meals'])

    # Similarly handle other fields like 'smoking_habit' and 'tracks_daily_calories'
    data['smoking_habit'] = int(data['smoking_habit'])
    data['tracks_daily_calories'] = int(data['tracks_daily_calories'])
    
    # Process 'alcohol_intake' safely without calling .lower() on non-string values
    if isinstance(data['alcohol_intake'], str):
        data['alcohol_intake'] = {'no': 0, 'sometimes': 1, 'frequently': 2, 'always': 3}[data['alcohol_intake'].lower()]
    else:
        data['alcohol_intake'] = int(data['alcohol_intake'])
    
    # Process 'transportation_used' safely
    if isinstance(data['transportation_used'], str):
        data['transportation_used'] = {'automobile': 1, 'motorbike': 2, 'bike': 3, 'public transportation': 4, 'walking': 5}[data['transportation_used'].lower()]
    else:
        data['transportation_used'] = int(data['transportation_used'])
    
    return pd.DataFrame([data])

# BMI Calculation
def calculate_bmi(height, weight):
    return weight / (height ** 2)

# Streamlit UI for user input
st.title('Obesity Prediction')

# Gender input
gender = st.selectbox('What is your gender?', ['Female', 'Male'])
age = st.number_input('What is your age?', min_value=14, max_value=61, value=25)
height_m = st.number_input('What is your height (in meters)?', min_value=1.45, max_value=1.98, value=1.7)
weight_kg = st.number_input('What is your weight (in kilograms)?', min_value=39.0, max_value=173.0, value=70.0)
family_history_with_overweight = st.selectbox('Has a family member suffered or suffers from overweight?', ['Yes', 'No'])
high_calorie_intake = st.selectbox('Do you eat high caloric food frequently?', ['Yes', 'No'])
vegetable_consumption = st.selectbox('Do you usually eat vegetables in your meals?', ['Never', 'Sometimes', 'Always'])
daily_meal_count = st.selectbox('How many main meals do you have daily?', ['Between 1 and 2', 'Three', 'More than three'])
food_between_meals = st.selectbox('Do you eat any food between meals?', ['No', 'Sometimes', 'Frequently', 'Always'])
smoking_habit = st.selectbox('Do you smoke?', ['Yes', 'No'])
water_consumption = st.selectbox('How much water do you drink daily?', ['Less than a liter', 'Between 1 and 2 L', 'More than 2 L'])
tracks_daily_calories = st.selectbox('Do you monitor the calories you eat daily?', ['Yes', 'No'])
exercise_frequency = st.selectbox('How often do you have physical activity?', ['I do not have', '1 or 2 days', '2 or 4 days', '4 or 5 days'])
tech_usage_time = st.selectbox('How much time do you use technological devices?', ['0–2 hours', '3–5 hours', 'More than 5 hours'])
alcohol_intake = st.selectbox('How often do you drink alcohol?', ['I do not drink', 'Sometimes', 'Frequently', 'Always'])
transportation_used = st.selectbox('Which transportation do you usually use?', ['Automobile', 'Motorbike', 'Bike', 'Public Transportation', 'Walking'])

# Calculate BMI
bmi = calculate_bmi(height_m, weight_kg)
st.write(f"Your BMI is: {bmi:.2f}")

# Collect input data into a dictionary
input_data = {
    'gender': gender,
    'age': age,
    'height_m': height_m,
    'weight_kg': weight_kg,
    'family_history_with_overweight': family_history_with_overweight == 'Yes',
    'high_calorie_intake': high_calorie_intake == 'Yes',
    'vegetable_consumption': {'Never': 1, 'Sometimes': 2, 'Always': 3}[vegetable_consumption],
    'daily_meal_count': {'Between 1 and 2': 1, 'Three': 2, 'More than three': 3}[daily_meal_count],
    'food_between_meals': food_between_meals,
    'smoking_habit': smoking_habit == 'Yes',
    'water_consumption': {'Less than a liter': 1, 'Between 1 and 2 L': 2, 'More than 2 L': 3}[water_consumption],
    'tracks_daily_calories': tracks_daily_calories == 'Yes',
    'exercise_frequency': {'I do not have': 0, '1 or 2 days': 1, '2 or 4 days': 2, '4 or 5 days': 3}[exercise_frequency],
    'tech_usage_time': {'0–2 hours': 0, '3–5 hours': 1, 'More than 5 hours': 2}[tech_usage_time],
    'alcohol_intake': {'I do not drink': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[alcohol_intake],
    'transportation_used': {'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public Transportation': 3, 'Walking': 4}[transportation_used]
}

# Preprocess the input data
processed_input = preprocess_input(input_data)

# Make prediction
if st.button('Predict Obesity Level'):
    prediction = model.predict(processed_input)
    st.write(f'Predicted Obesity Level: {prediction[0]}')

    # Additional message based on BMI
    if bmi < 18.5:
        st.write("Your BMI indicates that you are underweight. Consider consulting with a healthcare provider.")
    elif 18.5 <= bmi < 24.9:
        st.write("Your BMI indicates that you are in a healthy weight range. Keep maintaining a balanced lifestyle.")
    elif 25 <= bmi < 29.9:
        st.write("Your BMI indicates that you are overweight. It's recommended to monitor your diet and exercise.")
    else:
        st.write("Your BMI indicates that you are obese. Consider consulting with a healthcare provider for personalized advice.")
