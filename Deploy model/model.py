import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_survive')

# Function to preprocess input data
def preprocess_input(data):
    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Ensure all necessary columns are present
    necessary_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    for col in necessary_columns:
        if col not in data.columns:
            data[col] = 0
    return data[necessary_columns]

# Function to make predictions
def predict_survival(data):
    preprocessed_data = preprocess_input(data)

    return model.predict(preprocessed_data)

# Create the Streamlit app
def main():
    st.title('Titanic Survival Prediction')
    st.write('This app predicts the survival of passengers on the Titanic.')

    # User input for features
    Pclass = st.selectbox('Pclass', [1, 2, 3])
    Age = st.number_input('Age', min_value=0, max_value=100, value=30)
    SibSp = st.number_input('SibSp', min_value=0, max_value=8, value=0)
    Parch = st.number_input('Parch', min_value=0, max_value=6, value=0)
    Fare = st.number_input('Fare', min_value=0.0, max_value=600.0, value=30.0, step=0.01)
    Sex = st.selectbox('Sex', ['male', 'female'])
    Embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Pclass': [Pclass], 'Age': [Age], 'SibSp': [SibSp],
        'Parch': [Parch], 'Fare': [Fare], 'Sex': [Sex], 'Embarked': [Embarked]
    })
    
    # Make prediction
    if st.button('Predict'):
        prediction = predict_survival(input_data)
        if prediction[0] == 1:
            st.write('The passenger is predicted to survive.')
        else:
            st.write('The passenger is predicted not to survive.')
            
# Run the app
if __name__ == '__main__':
    main()
