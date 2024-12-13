import json

file_path = ''

with open(file_path, 'r') as file:
    notebook = json.load(file)

code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']

extracted_code = "\n".join(["".join(cell) for cell in code_cells])

with open('../coursework/models/models_training/train_model_libs.py', 'w') as code_file:
    code_file.write(extracted_code)

print(extracted_code)
