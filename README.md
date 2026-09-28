# Anti-Fraud Model for Internet Loans Using Machine Learning

A polished Streamlit academic prototype for suspicious online-loan application detection.

## Important
This version is intentionally packaged with **synthetic training data** so it runs immediately and does not require real customer information. It is a demonstration, not a real lending/fraud-decision system.

## Features
- Professional dark financial-security dashboard
- Loan application analysis form
- Random Forest fraud-risk classifier
- Isolation Forest anomaly detector
- Risk score and risk band
- Human-readable explanation factors
- Confusion matrix and model metrics
- Feature importance visualization
- Responsive Streamlit UI

## Windows setup

1. Install Python 3.11 or newer from python.org.
2. Open PowerShell in this folder.
3. Create a virtual environment:

```powershell
python -m venv .venv
```

4. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

then activate again.

5. Install packages:

```powershell
pip install -r requirements.txt
```

6. Start the application:

```powershell
streamlit run app.py
```

7. Open the local address Streamlit displays, usually:

http://localhost:8501

## Demo flow
Dashboard -> Analyze Application -> enter values -> Analyze Application -> inspect risk score, fraud probability, anomaly signal and explanations.

## Next development stage
Replace the synthetic training data with a suitable benchmark/approved dataset and add SHAP, model comparison, cross-validation, threshold tuning, audit logs, and a proper database.
