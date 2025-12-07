import joblib

clf = joblib.load("investment_classifier.pkl")
print(clf)

print("\nExpected columns in model:")
print(clf.named_steps['prep'].transformers)
