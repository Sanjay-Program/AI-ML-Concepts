from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

# Define meta-model
meta_model = LogisticRegression()

# Stacking model
model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred))
