# File: predict_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------------
# 1. Load the raw dataset to compute defaults and transformation parameters
# ----------------------------------------------------------------------------
df = pd.read_csv("insurance.csv")

# Compute default values from raw data
default_age = df['age'].median()
default_bmi = df['bmi'].median()
default_children = int(df['children'].median())  # you may choose mode if preferred
default_sex = df['sex'].mode()[0]       # 'male' or 'female'
default_smoker = df['smoker'].mode()[0]   # 'no' or 'yes'
default_region = df['region'].mode()[0]

# Compute normalization parameters for numeric features (as used in preprocess_split.py)
mean_age = df['age'].mean()
std_age  = df['age'].std()

mean_bmi = df['bmi'].mean()
std_bmi  = df['bmi'].std()

mean_children = df['children'].mean()
std_children  = df['children'].std()

# ----------------------------------------------------------------------------
# 2. Create categorical mappings exactly as in preprocess_split.py
# ----------------------------------------------------------------------------
# For sex: male=0, female=1
sex_map = {'male': 0, 'female': 1}

# For smoker: no=0, yes=1
smoker_map = {'no': 0, 'yes': 1}

# For region: use the order in which they first appear in the dataset
region_mapping = {region: i for i, region in enumerate(df['region'].unique())}
# Prepare a list of region names (in the same order as used in training)
region_list = list(region_mapping.keys())

# ----------------------------------------------------------------------------
# 3. Load the learned model parameters (theta_libs.txt) from the libs model
# ----------------------------------------------------------------------------
theta_libs = np.loadtxt("theta_libs.txt")
# NOTE: The model was trained on the numeric dataset where features were:
# [intercept, normalized age, encoded sex, normalized bmi, normalized children, encoded smoker, encoded region]

# ----------------------------------------------------------------------------
# 4. Define a prediction function that transforms raw inputs
# ----------------------------------------------------------------------------
def predict_charge(raw_age, raw_sex, raw_bmi, raw_children, raw_smoker, raw_region):
    """
    Given raw (real-world) inputs, this function:
      - Normalizes the numeric features using the training set parameters.
      - Label-encodes categorical inputs.
      - Forms the feature vector (with intercept) and applies the linear model.
      - Exponentiates the log-prediction to return a dollar value.
    """
    # Normalize numeric features
    norm_age = (raw_age - mean_age) / std_age
    norm_bmi = (raw_bmi - mean_bmi) / std_bmi
    norm_children = (raw_children - mean_children) / std_children

    # Map categorical features
    mapped_sex = sex_map[raw_sex.lower()]
    mapped_smoker = smoker_map[raw_smoker.lower()]
    mapped_region = region_mapping[raw_region]

    # Form feature vector (including intercept)
    # Feature order: [1, norm_age, mapped_sex, norm_bmi, norm_children, mapped_smoker, mapped_region]
    x = np.array([1.0, norm_age, mapped_sex, norm_bmi, norm_children, mapped_smoker, mapped_region], dtype=float)

    # Compute log-charge prediction then exponentiate to get charge in dollars
    log_pred = np.dot(theta_libs, x)
    return np.exp(log_pred)

# ----------------------------------------------------------------------------
# 5. Build the UI
# ----------------------------------------------------------------------------
root = tk.Tk()
root.title("Medical Charges Predictor")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

def create_labeled_entry(label_text, row, default_val):
    ttk.Label(frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=5)
    entry = ttk.Entry(frame)
    entry.insert(0, str(default_val))
    entry.grid(row=row, column=1, pady=5)
    return entry

# Create entry fields for raw numeric inputs
age_entry = create_labeled_entry("Возраст:", 0, default_age)
bmi_entry = create_labeled_entry("Индекс массы тела:", 1, default_bmi)
children_entry = create_labeled_entry("Кол-во детей:", 2, default_children)

# For categorical values, use Comboboxes populated with raw labels.
ttk.Label(frame, text="Пол:").grid(row=3, column=0, sticky=tk.W, pady=5)
sex_combo = ttk.Combobox(frame, values=["male", "female"], state="readonly")
sex_combo.set(default_sex.lower())
sex_combo.grid(row=3, column=1, pady=5)

ttk.Label(frame, text="Курильщик:").grid(row=4, column=0, sticky=tk.W, pady=5)
smoker_combo = ttk.Combobox(frame, values=["Да", "Нет"], state="readonly")
smoker_combo.set(default_smoker.lower())
smoker_combo.grid(row=4, column=1, pady=5)

ttk.Label(frame, text="Регион:").grid(row=5, column=0, sticky=tk.W, pady=5)
region_combo = ttk.Combobox(frame, values=region_list, state="readonly")
region_combo.set(default_region)
region_combo.grid(row=5, column=1, pady=5)

def on_predict():
    try:
        # Read raw inputs from UI
        raw_age = float(age_entry.get())
        raw_bmi = float(bmi_entry.get())
        raw_children = float(children_entry.get())
        raw_sex = sex_combo.get()
        raw_smoker = smoker_combo.get()
        raw_region = region_combo.get()

        # Get prediction (in dollars)
        prediction = predict_charge(raw_age, raw_sex, raw_bmi, raw_children, raw_smoker, raw_region)
        messagebox.showinfo("Предсказание", f"Предсказать медицинские расходы: ${prediction:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

predict_btn = ttk.Button(frame, text="Предсказать", command=on_predict)
predict_btn.grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
