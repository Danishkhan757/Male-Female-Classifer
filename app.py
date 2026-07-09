import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Dataset
df = pd.read_csv("gender.csv")   # columns: name, gender

df['gender'] = df['gender'].map({'Male':1,'Female':0})

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
X = vectorizer.fit_transform(df['name'])

X_train, X_test, y_train, y_test = train_test_split(
    X, df['gender'], test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, "gender_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Accuracy:", model.score(X_test, y_test))