import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pickle import load

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = load(open('DiseaseConditionPred_Model.pkl', 'rb'))
vectorizer = load(open('vectorizer.pkl', 'rb'))

# Function to clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# App title
st.title("Disease Condition Prediction App")

st.write("Enter patient review details to predict the associated medical condition:")

# User inputs
review_text = st.text_area("Patient Review")

# When Predict button is clicked
if st.button("Predict Condition"):
    if review_text.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        # Clean and transform the review
        cleaned_review = clean_text(review_text)
        review_vector = vectorizer.transform([cleaned_review])
        
        # Predict using model
        prediction = model.predict(review_vector)[0]

        # Prepare result DataFrame
        result_df = pd.DataFrame({
            'review': [review_text],
            'condition_predicted': [prediction]
        })

        # Show result
        st.success("Prediction complete!")
        st.dataframe(result_df)