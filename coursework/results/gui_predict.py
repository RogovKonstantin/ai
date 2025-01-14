"""
File: gui_predict.py

A simple Tkinter GUI that:
1) Prompts for each of the 15 features from your dataset:
   - 8 numeric features that need standardization
   - 7 categorical features (e.g., 0 or 1).
2) Applies the same standard scaling (using stored means & stds).
3) Loads manual logistic regression weights from model_manual_weights.txt.
4) Displays predicted TenYearCHD (0 or 1).
"""

import tkinter as tk
import pickle
import math

# 1) We'll define your numerical & categorical features.
numerical_features = ['age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose']
categorical_features = ['male','education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes']
all_features = numerical_features + categorical_features

# 2) Load your precomputed means & scales from a pickle file (or you could hard-code them).
#    This file was presumably saved during "preprocess_and_save()" or after you fit your StandardScaler.
try:
    scaler_params = pickle.load(open("../datasets/datasets_processing/scaler_params.pkl", "rb"))
    means = scaler_params["means"]  # list of shape [8]
    scales = scaler_params["scales"]  # list of shape [8]
except Exception as e:
    print("Error loading scaler parameters:", e)
    print("Be sure to place 'scaler_params.pkl' in the same folder or adjust the path.")
    means = [0]*8  # fallback, but won't produce correct scaling
    scales = [1]*8

# 3) Load the manual logistic regression weights
def load_manual_weights(file_path):
    with open(file_path, 'r') as f:
        weights_str = f.read().strip().split(',')
    return list(map(float, weights_str))

# Sigmoid for the manual prediction
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Manual predict function
def predict_manual(inputs, weights):
    """
    inputs: an array of length (15) with already standardized numeric features
            and unmodified categorical features at the end.
    weights: array of length 16 (1 bias + 15 features)
    """
    # Add bias term
    x_with_bias = [1.0] + inputs
    # Dot product
    z = sum(w*x for w,x in zip(weights, x_with_bias))
    prob = sigmoid(z)
    return 1 if prob >= 0.5 else 0

# GUI Class
class HeartDiseasePredictorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("TenYearCHD Prediction (Manual Model)")

        # We'll store the input Entry widgets in a dictionary
        self.entries = {}

        row_index = 0

        tk.Label(master, text="Enter numeric features (unscaled):").grid(row=row_index, column=0, columnspan=2, pady=5, sticky="w")

        row_index += 1
        for feat in numerical_features:
            tk.Label(master, text=f"{feat}:").grid(row=row_index, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(master)
            entry.grid(row=row_index, column=1, padx=5, pady=2)
            self.entries[feat] = entry
            row_index += 1

        tk.Label(master, text="Enter categorical features (0 or 1, or numeric for 'education'):").grid(row=row_index, column=0, columnspan=2, pady=5, sticky="w")
        row_index += 1
        for feat in categorical_features:
            tk.Label(master, text=f"{feat}:").grid(row=row_index, column=0, padx=5, pady=2, sticky="e")
            entry = tk.Entry(master)
            entry.grid(row=row_index, column=1, padx=5, pady=2)
            self.entries[feat] = entry
            row_index += 1

        # Predict button
        self.predict_button = tk.Button(master, text="Predict CHD", command=self.on_predict)
        self.predict_button.grid(row=row_index, column=0, columnspan=2, pady=10)

        row_index += 1
        # Result label
        self.result_label = tk.Label(master, text="", font=("Helvetica", 12, "bold"))
        self.result_label.grid(row=row_index, column=0, columnspan=2, pady=10)

        # Load the manual weights
        self.manual_weights = load_manual_weights("../models/model_manual_weights.txt")

    def on_predict(self):
        """
        Gathers all user input, applies standard scaling to numeric features,
        leaves categorical features as is, then uses manual logistic regression
        weights to predict TenYearCHD.
        """
        # 1) Get the user input from Entry widgets
        user_input_numeric = []
        for i, feat in enumerate(numerical_features):
            val_str = self.entries[feat].get().strip()
            try:
                val_float = float(val_str)
            except:
                val_float = 0.0
            user_input_numeric.append(val_float)

        user_input_categorical = []
        for feat in categorical_features:
            val_str = self.entries[feat].get().strip()
            try:
                val_float = float(val_str)
            except:
                val_float = 0.0
            user_input_categorical.append(val_float)

        # 2) Standardize the numeric features using means/scales
        #    numeric z = (val - mean) / scale
        standardized_numeric = []
        for i, val in enumerate(user_input_numeric):
            z = (val - means[i]) / scales[i]  # replicate the StandardScaler
            standardized_numeric.append(z)

        # 3) Combine standardized numeric + raw categorical
        all_inputs = standardized_numeric + user_input_categorical

        # 4) Predict with manual logistic regression
        prediction = predict_manual(all_inputs, self.manual_weights)

        # 5) Show result
        if prediction == 1:
            msg = "Prediction: CHD within 10 years (1)"
        else:
            msg = "Prediction: No CHD within 10 years (0)"

        self.result_label.config(text=msg)


if __name__ == "__main__":
    root = tk.Tk()
    gui = HeartDiseasePredictorGUI(root)
    root.mainloop()
