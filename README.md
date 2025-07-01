# Disease_Prediction_Using_Review_Data

## 📌 Objective

To develop an NLP-based classification model that predicts the **effectiveness of a drug** (rating or condition) based on patient reviews using:

- Text Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Extraction (TF-IDF, CountVectorizer)
- Model Training (Logistic Regression, Naive Bayes, etc.)
- Evaluation Metrics (Accuracy, F1 Score, Confusion Matrix)

## 📊 Dataset

- **Source**: Drug Review Dataset (Drugs.com)
- **Features**:
  - `drugName`: Name of the drug
  - `condition`: Medical condition the drug is used for
  - `review`: User review in text format
  - `rating`: Numerical rating (1-10)
  - `date`: Date of review
  - `usefulCount`: Number of users who found the review helpful

## 🔧 Technologies Used

- Python 🐍
- NLP Libraries: `nltk`, `re`, `scikit-learn`
- Visualization: `matplotlib`, `seaborn`, `wordcloud`
- Machine Learning Models: Logistic Regression, Naive Bayes, SVM

## ⚙️ Steps Performed

1. **Data Cleaning**:
   - Removed nulls and duplicate entries
   - Cleaned review text (lowercasing, removing punctuation, stopwords)

2. **EDA**:
   - Word clouds for popular drugs and conditions
   - Distribution of ratings and word frequency

3. **Text Vectorization**:
   - TF-IDF and Bag of Words

4. **Modeling**:
   - Trained Logistic Regression, Naive Bayes, SVM classifiers
   - Evaluated using accuracy, confusion matrix, classification report

5. **Conclusion**:
   - Highlighted best-performing model
   - Discussed potential improvements (e.g., LSTM, BERT)

## 📈 Model Performance

| Model               | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 86%      | 0.84     |


# Save the content to a file
readme_path = "/mnt/data/README_Drug_Prediction.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

