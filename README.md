# bit-intro-assignment
My assignment for an interview at a lovely company Bit :)

**NOTE FOR RUNNING:**  

This project uses **uv** as the package manager and requires **Python 3.12+**. To set up the environment, run the following commands:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux / MacOS:
source .venv/bin/activate

# Install uv package manager
pip install uv

# Sync all required packages into the virtual environment
uv syncoost 

# Inference example
python main.py --inference --model lgbm --inference_model_path final_lgbm_model.txt --data_point data/inference/test_row.csv

# Training example
python main.py --training --model catboost
```

This should work. In case of questions, email me!