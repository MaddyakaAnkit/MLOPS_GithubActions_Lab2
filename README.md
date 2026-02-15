# GitHub Actions for ML Model Training, Evaluation, and Versioning

This repository demonstrates how to automate machine learning model training, evaluation, calibration, and versioning using **GitHub Actions**. The project uses **Logistic Regression** on synthetic datasets for demonstration and can be extended to real datasets.

---

## **Table of Contents**

1. [Project Overview]  
2. [Features]
3. [Prerequisites] 
4. [Project Structure] 
5. [Getting Started] 
6. [Running Locally]  
7. [GitHub Actions Workflow]  
8. [Logs and Results]
9. [License] 

---

## **Project Overview**

This project automates the lifecycle of a machine learning model:

1. **Training**: Logistic Regression model is trained on synthetic datasets.  
2. **Evaluation**: Accuracy and F1 Score metrics are automatically calculated.  
3. **Versioning**: Models are saved with timestamp-based versions in the `models/` folder.  
4. **Calibration**: Model probabilities can be calibrated to improve reliability.  
5. **Automation**: GitHub Actions handle retraining, evaluation, and versioning on schedule or push to `main`.

---

## **Features**

- Train a **Logistic Regression model**.  
- Automatically log metrics using **MLflow**.  
- Save models in `models/` folder with timestamps.  
- Save evaluation metrics in `metrics/` folder.  
- GitHub Actions workflows:
  - `model_calibration.yml` → scheduled calibration  
  - `model_calibration_on_push.yml` → triggers on push to main branch  

---

## **Prerequisites**

- Python 3.9+  
- Git  
- Virtual environment (recommended)  
- Required Python packages:
  ```bash
  pip install -r requirements.txt


Requirements include:

scikit-learn

mlflow

joblib

## **Project Structure**
Lab2-GitHub-Actions-ML/      <-- Repository root
│
├── src/                     <-- Source code
│   ├── train_model.py       <-- Training script (Logistic Regression)
│   ├── evaluate_model.py    <-- Evaluation script
│   └── test.ipynb           <-- Optional Jupyter notebook for testing
│
├── models/                  <-- Trained models (added after running train_model.py)
│   └── model_<timestamp>_lr_model.joblib
│
├── metrics/                 <-- Evaluation metrics (JSON)
│   └── <timestamp>_metrics.json
│
├── .github/                 <-- GitHub Actions workflows
│   └── workflows/
│       ├── model_calibration.yml
│       └── model_calibration_on_push.yml
│
├── data/                    <-- Optional: saved datasets
│   ├── data.pickle
│   └── target.pickle
│
├── requirements.txt         <-- Python dependencies (scikit-learn, mlflow, etc.)
├── README.md                <-- Project README
└── .gitignore               <-- Ignore files like __pycache__, .ipynb_checkpoints, etc.



## **Getting Started**

1. Clone this repository
git clone https://github.com/<your-username>/<your-repo>.git
cd Lab2

2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

3. Install dependencies
pip install -r requirements.txt

## **Running Locally**

1. Train the model
python src/train_model.py --timestamp "20251006120000"

2. Evaluate the model
python src/evaluate_model.py --timestamp "20251006120000"

3. Check results
Trained model is saved in: models/
Evaluation metrics are saved in: metrics/

# VS Code Execution Screenshots
![alt text](<Screenshot 2025-10-06 191839.png>)
![alt text](<Screenshot 2025-10-06 191854.png>)
![](<Screenshot 2025-10-06 192024.png>)


## **GitHub Actions Workflow**
1. Scheduled Model Calibration

Workflow: model_calibration.yml
Runs on a daily schedule (cron job)
Steps:
    Checkout repository
    Setup Python
    Install dependencies
    Generate timestamp
    Retrain and calibrate the model
    Save model and metrics
    Commit and push changes

2. Retraining on Push

Workflow: model_calibration_on_push.yml
Trigger: Any push to main branch
Performs the same steps as above automatically.

## **Logs and Results**

After running scripts locally or via GitHub Actions:
1. Models Folder
models/
  model_20251006120000_lr_model.joblib
2. Metrics Folder
metrics/
  20251006120000_metrics.json
3. Key takeaways
Phase    	Accuracy	   F1 Score	                     Notes
Training	   0.788	      0.785	             Model learned the training data.
Evaluation	0.58875	   0.58089	 Model tested on unseen synthetic data — shows generalization.
