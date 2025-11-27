import joblib, sys
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=42)

# Train model
model = LinearRegression().fit(X_tr, y_tr)
y_pred = model.predict(X_te)
score = r2_score(y_te, y_pred)

# Save model
joblib.dump(model, "linear_model.pkl")

# Validation (TODO complete)
if score < 0.4:
    print(f"FAIL: R2 {score:.3f} < 0.4")
    sys.exit(1)
print(f"PASS: R2 {score:.3f}")
