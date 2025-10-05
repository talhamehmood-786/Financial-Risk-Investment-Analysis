
💰 Credit Risk & Investment Analysis Dashboard
Developed in Python using Streamlit
🧠 Project Overview

This project combines Credit Risk Modeling and Investment Evaluation in a single, interactive web dashboard.
It allows users to:

Analyze borrower default risk using Logistic Regression

Compute Expected Loss (EL) for financial loans

Evaluate investment profitability through IRR (Internal Rate of Return) and NPV (Net Present Value)

Receive automated financial recommendations for banking, stock, and loan investment strategies

⚙️ Technical Summary
Category	Details
Programming Language	Python
Framework	Streamlit
Libraries Used	pandas, numpy, scikit-learn, seaborn, matplotlib, numpy-financial
Model Type	Logistic Regression
Visualization Tools	Seaborn, Matplotlib
Deployment Option	Streamlit Cloud (Free Hosting)
📊 1. Credit Risk Analysis

The first module predicts whether a borrower will default (1) or not default (0) based on financial and demographic data.

Key Features:

Upload your own credit dataset (CSV format)

Model automatically trains using Logistic Regression

Visual outputs:

Confusion Matrix

ROC Curve and AUC Score

Distribution of Probability of Default (PD)

Calculates Expected Loss (EL) for each customer using the formula:

𝐸
𝐿
=
𝑃
𝐷
×
𝐸
𝐴
𝐷
×
𝐿
𝐺
𝐷
EL=PD×EAD×LGD

Where:

PD = Probability of Default (predicted by the model)

EAD = Exposure at Default (loan amount)

LGD = Loss Given Default (user-adjustable slider)

✅ The output includes a downloadable table with PD and EL for each loan.

💼 2. Investment Analysis (IRR & NPV)

The second module helps users evaluate investment options such as stocks, bank deposits, or loan portfolios.

Features:

Input initial investment, discount rate, and yearly cash flows

Calculates:

IRR (Internal Rate of Return) — profitability of the investment

NPV (Net Present Value) — total value of future returns today

Visualizes cash flows with an interactive bar chart

Provides investment recommendations:

IRR > 15% → Stock investments are more attractive

IRR between 8–15% → Bank deposits or balanced funds are suitable

IRR < 8% → Safer fixed deposits or government bonds preferred

🧮 Sample Dataset

A sample CSV is included with columns:

income, loan_amount, credit_score, age, default


Each record represents an individual loan applicant with associated financial data.

🚀 How to Run

Install requirements:

pip install streamlit pandas numpy scikit-learn seaborn matplotlib numpy-financial


Run the app:

streamlit run app.py


Open your browser at http://localhost:8501

🌐 Deployment

You can easily deploy your app for free using Streamlit Cloud
.
Once deployed, you’ll receive a public link (e.g., https://yourname-credit-risk.streamlit.app) to share with instructors or clients.

🎯 Learning Outcomes

Understand logistic regression in credit risk modeling

Apply IRR and NPV in evaluating investment projects

Visualize and interpret financial metrics interactively

Develop and deploy a real-time financial analytics web app

