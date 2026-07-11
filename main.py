import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the data
data = pd.read_csv('SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])

print(data.head())
print(data['label'].value_counts())

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

data['clean_message'] = data['message'].apply(clean_text)
print(data[['message', 'clean_message']].head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training data shape:", X_train_vec.shape)
print("Testing data shape:", X_test_vec.shape)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed report:\n", classification_report(y_test, y_pred))

# Interactive testing
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

while True:
    user_input = input("\nEnter a message to check (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    result = predict_message(user_input)
    print(f"Prediction: {result.upper()}")