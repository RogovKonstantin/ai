# File: model_libs.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

# -----------------------------
# 1. Load numeric train/test data (all numeric, log-target)
# -----------------------------
train_df = pd.read_csv("train_numeric.csv")
test_df  = pd.read_csv("test_numeric.csv")

X_train = train_df.drop("charges", axis=1)
y_train = train_df["charges"]  # log-transformed
X_test  = test_df.drop("charges", axis=1)
y_test  = test_df["charges"]

# -----------------------------
# 2. Fit scikit-learn LinearRegression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

# -----------------------------
# 3. Compute MSE & R² in log-space
# -----------------------------
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test  = r2_score(y_test, y_pred_test)

print("Libs Model (LinearRegression)")
print("Test MSE:", mse_test)
print("Test R²:", r2_test)

# -----------------------------
# 4. Plot Predicted vs. Actual (log-space)
# -----------------------------
plt.figure()
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Фактические значения")
plt.ylabel("Предсказанные значения")
plt.title("Предсказанные против фактических")
plt.savefig("predicted_vs_actual_libs.png")
plt.close()

# -----------------------------
# 5. Generate a Learning Curve
# -----------------------------
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    scoring="neg_mean_squared_error",
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

# Convert negative MSE to positive MSE
train_mse = -np.mean(train_scores, axis=1)
val_mse   = -np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mse, 'o-', label="Training MSE (log-space)")
plt.plot(train_sizes, val_mse, 'o-', label="Validation MSE (log-space)")
plt.xlabel("Размер датасета")
plt.ylabel("MSE")
plt.title("Learning Curve")
plt.legend()
plt.savefig("libs_learning_curve.png")
plt.close()

# -----------------------------
# 6. Save learned parameters
# -----------------------------
theta_libs = np.hstack([model.intercept_, model.coef_])
np.savetxt("theta_libs.txt", theta_libs)
print("Saved libs model parameters to theta_libs.txt.")
print("Plots saved: predicted_vs_actual_libs.png, libs_learning_curve.png")
