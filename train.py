import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = {
    'age': [25, 45, 35, 52, 23, 40, 60, 48, 33, 55],
    'income': [50000, 100000, 70000, 120000, 40000, 90000, 150000, 110000, 65000, 130000],
    'years_employed': [1, 20, 10, 25, 0, 15, 30, 22, 8, 28],
    'children': [0, 2, 1, 3, 0, 2, 4, 3, 1, 3],
    'approved': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['age', 'income', 'years_employed', 'children']]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'model.pkl')

