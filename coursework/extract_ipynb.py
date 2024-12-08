import json

# Path to the .ipynb file
file_path = 'C:/Users/Kostya/Desktop/heart-disease-prediction.ipynb'

# Read the notebook as a JSON file
with open(file_path, 'r') as file:
    notebook = json.load(file)

# Extract code from code cells
code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']

# Join all code cells into a single string
extracted_code = "\n".join(["".join(cell) for cell in code_cells])

# Optionally, save the extracted code to a .py file
with open('extracted_code.py', 'w') as code_file:
    code_file.write(extracted_code)

print(extracted_code)  # Prints the extracted code
