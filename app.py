# app.py — Credit Risk, Investment & Risk Analysis Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import numpy_financial as npf

# --- Streamlit Page Config ---
st.set_page_config(page_title="Financial Risk & Investment Dashboard", layout="wide")
st.title("💰 Financial Risk & Investment Analysis Dashboard")

st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Select a Section:", ["Credit Risk Analysis", "Investment Analysis"])

# =====================================================
# === CREDIT RISK ANALYSIS SECTION ====================
# =====================================================
if page == "Credit Risk Analysis":
    st.header("🏦 Credit Risk Analysis using")

    uploaded_file = st.file_uploader("📂 Upload your credit dataset (CSV)", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("✅ Data Loaded Successfully")
        st.write("### Data Preview", data.head())

        data.dropna(inplace=True)

        if 'default' not in data.columns:
            st.error("❌ The dataset must contain a 'default' column (0 = no default, 1 = default).")
            st.stop()

        # Features and Target
        X = data.drop(columns=['default'])
        y = data['default']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluation
        st.subheader("📊 Model Evaluation")
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        auc_score = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax2.plot([0,1],[0,1],'--',color='gray')
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)

        # Expected Loss
        st.subheader("💸 Expected Loss Calculation")
        LGD = st.slider("Select Loss Given Default (LGD)", 0.0, 1.0, 0.5, 0.05)
        if 'loan_amount' not in X_test.columns:
            st.error("❌ Column 'loan_amount' not found! Please include it in your dataset.")
            st.stop()

        EAD = X_test['loan_amount']
        PD = y_prob
        expected_loss = PD * EAD * LGD

        results = X_test.copy()
        results['PD'] = PD
        results['EL'] = expected_loss
        results['Actual_Default'] = y_test.values

        st.write("### Sample of Expected Loss Calculations")
        st.dataframe(results[['loan_amount', 'PD', 'EL']].head(10))

        fig3, ax3 = plt.subplots()
        sns.histplot(PD, bins=20, kde=True, color='purple', ax=ax3)
        ax3.set_title("Distribution of Predicted Probability of Default (PD)")
        ax3.set_xlabel("Probability of Default")
        st.pyplot(fig3)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Results CSV",
            data=csv,
            file_name='credit_risk_results.csv',
            mime='text/csv'
        )

    else:
        st.info("👆 Upload a CSV file to begin analysis.")
        st.markdown("""
        **Expected columns:**
        - income  
        - loan_amount  
        - credit_score  
        - age  
        - default (0 = no default, 1 = default)
        """)

# =====================================================
# === INVESTMENT ANALYSIS SECTION =====================
# =====================================================
elif page == "Investment Analysis":
    st.header("📈 Investment Plan Evaluation (NPV & IRR)")

    st.write("This tool helps you compare **bank**, **stock**, or **loan-based** investments using NPV and IRR metrics.")

    st.subheader("💡 Enter Investment Details")

    initial_investment = st.number_input("💰 Initial Investment (negative number, e.g. -10000)", value=-10000.0)
    years = st.number_input("📆 Number of Years", value=5, step=1)
    discount_rate = st.slider("📉 Discount Rate (as %)", 0.0, 30.0, 10.0) / 100

    st.write("### 📤 Enter Annual Cash Flows (Positive Values)")
    cash_flows = []
    for i in range(int(years)):
        cf = st.number_input(f"Year {i+1} Cash Flow", value=3000.0, key=f"cf_{i}")
        cash_flows.append(cf)

    if st.button("Calculate IRR & NPV"):
        all_flows = [initial_investment] + cash_flows
        irr = npf.irr(all_flows)
        npv = npf.npv(discount_rate, all_flows)

        st.success(f"✅ **IRR:** {irr*100:.2f}%")
        st.info(f"💵 **NPV:** {npv:,.2f}")

        if irr > discount_rate:
            st.markdown("### ✅ Investment is **Profitable** — IRR exceeds your discount rate.")
        else:
            st.markdown("### ⚠️ Investment is **Not Attractive** — IRR is below the discount rate.")

        st.write("### 🔍 Financial Advice:")
        if irr > 0.15:
            st.success("📈 High IRR (>15%) → Stock investments are more attractive.")
        elif irr > 0.08:
            st.info("🏦 Moderate IRR (8–15%) → Bank deposits or balanced mutual funds are reasonable.")
        else:
            st.warning("💤 Low IRR (<8%) → Prefer safe fixed deposits or government bonds.")

        st.write("### 💹 Cash Flow Visualization")
        fig, ax = plt.subplots()
        ax.bar(range(0, len(all_flows)), all_flows, color='teal')
        ax.set_title("Cash Flow Timeline")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cash Flow")
        st.pyplot(fig)
