
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Anti-Fraud AI | Internet Loans",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
:root {
    --bg: #07111f;
    --panel: #0d1b2d;
    --panel2: #10243b;
    --text: #eaf2ff;
    --muted: #91a4bd;
    --accent: #53b7ff;
    --good: #27d17f;
    --warn: #ffbf47;
    --bad: #ff5d73;
}
.stApp {
    background: linear-gradient(135deg, #06101d 0%, #0a1626 55%, #07111f 100%);
    color: var(--text);
}
.block-container { padding-top: 1.4rem; }
h1,h2,h3 { color: var(--text); }
.small-muted { color: #91a4bd; font-size: 0.92rem; }
.hero {
    padding: 1.3rem 1.5rem;
    border: 1px solid #1e3957;
    border-radius: 18px;
    background: linear-gradient(135deg, #0c2036, #0a1728);
    box-shadow: 0 12px 40px rgba(0,0,0,.22);
    margin-bottom: 1rem;
}
.hero-title { font-size: 2.25rem; font-weight: 800; letter-spacing: -.03em; }
.hero-sub { color: #9eb3cb; margin-top: .35rem; }
.card {
    padding: 1rem 1.1rem;
    border: 1px solid #1b3653;
    border-radius: 15px;
    background: rgba(13,27,45,.86);
}
.risk-high {
    border: 1px solid #8d3042;
    background: linear-gradient(135deg, #2a121c, #170f17);
}
.risk-medium {
    border: 1px solid #8b6a2d;
    background: linear-gradient(135deg, #2a2112, #17140d);
}
.risk-low {
    border: 1px solid #216e4a;
    background: linear-gradient(135deg, #10291f, #0c1915);
}
.badge {
    display:inline-block; padding:.3rem .65rem; border-radius:999px;
    font-weight:700; font-size:.82rem;
}
.badge-high { background:#5c1e2b; color:#ffb2be; }
.badge-medium { background:#59451d; color:#ffe09b; }
.badge-low { background:#174b36; color:#a8ffd3; }
div[data-testid="stMetricValue"] { color: #eef6ff; }
section[data-testid="stSidebar"] { background: #071525; border-right: 1px solid #17304a; }
.stButton > button {
    border-radius: 11px;
    border: 1px solid #2a75a8;
    background: linear-gradient(90deg, #126aa2, #1b83c4);
    color: white;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Synthetic training dataset
# ---------------------------
FEATURES = [
    "age", "annual_income", "employment_years", "loan_amount",
    "loan_term_months", "existing_loans", "monthly_debt",
    "credit_score", "previous_defaults", "recent_inquiries",
    "applications_30d", "address_years", "device_risk"
]

@st.cache_resource
def train_models(seed=42, n=12000):
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, n)
    annual_income = np.clip(rng.lognormal(np.log(550000), 0.55, n), 150000, 5000000)
    employment_years = np.clip(rng.gamma(2.2, 3.2, n), 0, 30)
    loan_amount = np.clip(rng.lognormal(np.log(450000), 0.75, n), 50000, 5000000)
    loan_term = rng.choice([6,12,18,24,36,48,60], n, p=[.04,.16,.12,.22,.25,.14,.07])
    existing_loans = rng.poisson(1.3, n)
    monthly_debt = np.clip(rng.lognormal(np.log(10000), .7, n), 0, 150000)
    credit_score = np.clip(rng.normal(680, 75, n), 300, 850)
    previous_defaults = rng.poisson(.25, n)
    recent_inquiries = rng.poisson(1.3, n)
    applications_30d = rng.poisson(.8, n)
    address_years = np.clip(rng.gamma(2.2, 2.5, n), .1, 30)
    device_risk = np.clip(rng.beta(1.8, 6, n), 0, 1)

    dti = monthly_debt / np.maximum(annual_income / 12, 1)
    loan_income = loan_amount / np.maximum(annual_income, 1)

    # Synthetic fraud propensity for a demonstration dataset.
    # This is intentionally not a real lending model and should not be used for
    # production lending decisions.
    logit = (
        -4.0
        + 1.8 * np.clip(dti, 0, 2)
        + 2.0 * np.clip(loan_income, 0, 12)
        + 0.38 * previous_defaults
        + 0.22 * recent_inquiries
        + 0.34 * applications_30d
        + 2.4 * device_risk
        - 0.002 * (credit_score - 500)
        - 0.035 * employment_years
        - 0.02 * address_years
        + 0.10 * existing_loans
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -20, 20)))
    y = (rng.random(n) < probability).astype(int)

    X = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term,
        "existing_loans": existing_loans,
        "monthly_debt": monthly_debt,
        "credit_score": credit_score,
        "previous_defaults": previous_defaults,
        "recent_inquiries": recent_inquiries,
        "applications_30d": applications_30d,
        "address_years": address_years,
        "device_risk": device_risk
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    iso = IsolationForest(
        n_estimators=250,
        contamination=0.06,
        random_state=42
    )
    iso.fit(X_train)

    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion": confusion_matrix(y_test, pred)
    }

    return clf, iso, X, y, metrics

clf, iso, train_X, train_y, metrics = train_models()

def money(x):
    return f"₹{x:,.0f}"

def assess(row):
    x = pd.DataFrame([row])[FEATURES]
    fraud_prob = float(clf.predict_proba(x)[0, 1])
    anomaly = int(iso.predict(x)[0] == -1)

    # Combine supervised probability and anomaly signal for a dashboard risk score.
    risk_score = min(99, round(100 * (0.82 * fraud_prob + 0.18 * anomaly)))

    if risk_score >= 70:
        band = "HIGH"
    elif risk_score >= 40:
        band = "MEDIUM"
    else:
        band = "LOW"

    dti = row["monthly_debt"] / max(row["annual_income"] / 12, 1)
    loan_income = row["loan_amount"] / max(row["annual_income"], 1)

    factors = []
    if loan_income > 6:
        factors.append(("High loan-to-income ratio", "negative"))
    elif loan_income > 4:
        factors.append(("Elevated loan-to-income ratio", "warning"))
    else:
        factors.append(("Loan-to-income ratio within model baseline", "positive"))

    if dti > .55:
        factors.append(("High debt-to-income pressure", "negative"))
    elif dti > .35:
        factors.append(("Moderate debt-to-income pressure", "warning"))
    else:
        factors.append(("Debt-to-income pressure is lower", "positive"))

    if row["applications_30d"] >= 4:
        factors.append(("Many recent loan applications", "negative"))
    elif row["applications_30d"] >= 2:
        factors.append(("Several recent applications", "warning"))
    else:
        factors.append(("Low recent application activity", "positive"))

    if row["previous_defaults"] >= 2:
        factors.append(("Multiple previous defaults", "negative"))
    elif row["previous_defaults"] == 1:
        factors.append(("Previous default recorded", "warning"))
    else:
        factors.append(("No previous defaults recorded", "positive"))

    if row["device_risk"] >= .65:
        factors.append(("Elevated device-risk signal", "negative"))
    elif row["device_risk"] >= .4:
        factors.append(("Moderate device-risk signal", "warning"))
    else:
        factors.append(("Lower device-risk signal", "positive"))

    if row["credit_score"] >= 720:
        factors.append(("Higher credit-score range", "positive"))
    elif row["credit_score"] < 580:
        factors.append(("Lower credit-score range", "negative"))

    action = {
        "HIGH": "MANUAL VERIFICATION REQUIRED",
        "MEDIUM": "ENHANCED REVIEW RECOMMENDED",
        "LOW": "NO HIGH-RISK PATTERN DETECTED"
    }[band]

    return fraud_prob, anomaly, risk_score, band, factors, action, dti, loan_income

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 🛡️ Anti-Fraud AI")
    st.caption("Internet Loan Security Platform")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Application", "Model Performance", "About"]
    )
    st.divider()
    st.caption("Demo model • Synthetic training data")
    st.caption("For academic demonstration only")

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="hero">
<div class="hero-title">🛡️ Anti-Fraud Model for Internet Loans</div>
<div class="hero-sub">Machine-learning assisted fraud-risk screening for online loan applications</div>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications analyzed", "1,248")
    c2.metric("Flagged for review", "86")
    c3.metric("Verified", "1,162")
    c4.metric("Model ROC-AUC", f"{metrics['roc_auc']:.2f}")

    st.markdown("### Security Overview")
    a, b = st.columns(2)

    with a:
        demo = pd.DataFrame({
            "Category": ["Low risk", "Medium risk", "High risk"],
            "Applications": [892, 270, 86]
        })
        fig = px.bar(demo, x="Category", y="Applications", text="Applications")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce9f8",
            margin=dict(l=10,r=10,t=20,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with b:
        labels = ["Verified", "Review", "Flagged"]
        values = [1162, 42, 44]
        fig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.58)])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#dce9f8",
            margin=dict(l=10,r=10,t=20,b=10),
            showlegend=True
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Recent Applications")
    recent = pd.DataFrame({
        "Application": ["APP-1024","APP-1025","APP-1026","APP-1027","APP-1028"],
        "Risk": ["LOW","HIGH","MEDIUM","LOW","HIGH"],
        "Status": ["Verified","Review Required","Investigation","Verified","Review Required"]
    })
    st.dataframe(recent, use_container_width=True, hide_index=True)

elif page == "Analyze Application":
    st.markdown("### 🔍 New Loan Analysis")
    st.caption("Enter a synthetic/demo application. The model produces a fraud-risk assessment, not a final lending decision.")

    with st.form("loan_form"):
        st.markdown("#### Applicant & Employment")
        a,b,c,d = st.columns(4)
        age = a.number_input("Age", 21, 65, 29)
        income = b.number_input("Annual income (₹)", 100000, 10000000, 600000, step=25000)
        emp = c.number_input("Employment years", 0.0, 40.0, 3.0, step=.5)
        address = d.number_input("Address stability (years)", .1, 40.0, 3.0, step=.5)

        st.markdown("#### Loan & Financial Profile")
        a,b,c,d = st.columns(4)
        loan = a.number_input("Loan amount (₹)", 25000, 10000000, 500000, step=25000)
        term = b.selectbox("Loan term (months)", [6,12,18,24,36,48,60], index=4)
        existing = c.number_input("Existing loans", 0, 15, 1)
        debt = d.number_input("Monthly debt (₹)", 0, 500000, 12000, step=1000)

        st.markdown("#### Credit & Application Signals")
        a,b,c,d,e = st.columns(5)
        credit = a.slider("Credit score", 300, 850, 690)
        defaults = b.number_input("Previous defaults", 0, 10, 0)
        inquiries = c.number_input("Recent credit inquiries", 0, 20, 1)
        apps = d.number_input("Applications in 30 days", 0, 20, 1)
        device = e.slider("Device-risk signal", 0.0, 1.0, .20, .01)

        submitted = st.form_submit_button("🛡️ ANALYZE APPLICATION", use_container_width=True)

    if submitted:
        row = {
            "age": age, "annual_income": income, "employment_years": emp,
            "loan_amount": loan, "loan_term_months": term,
            "existing_loans": existing, "monthly_debt": debt,
            "credit_score": credit, "previous_defaults": defaults,
            "recent_inquiries": inquiries, "applications_30d": apps,
            "address_years": address, "device_risk": device
        }
        fraud_prob, anomaly, risk_score, band, factors, action, dti, loan_income = assess(row)

        cls = {"HIGH":"risk-high","MEDIUM":"risk-medium","LOW":"risk-low"}[band]
        badge = {"HIGH":"badge-high","MEDIUM":"badge-medium","LOW":"badge-low"}[band]

        st.markdown("### AI Risk Assessment")
        left,right = st.columns([1,1.5])

        with left:
            st.markdown(f"""
            <div class="card {cls}">
                <div class="small-muted">OVERALL RISK SCORE</div>
                <div style="font-size:3rem;font-weight:800;">{risk_score}/100</div>
                <span class="badge {badge}">{band} ATTENTION</span>
                <hr style="border-color:#24405d">
                <div><b>Fraud model probability:</b> {fraud_prob*100:.1f}%</div>
                <div><b>Anomaly signal:</b> {"Detected" if anomaly else "Not detected"}</div>
                <div><b>Debt-to-income:</b> {dti*100:.1f}%</div>
                <div><b>Loan-to-income:</b> {loan_income:.2f}×</div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("#### Why did the model flag this application?")
            for text, kind in factors:
                icon = {"negative":"🔴","warning":"🟠","positive":"🟢"}[kind]
                st.markdown(f"{icon} **{text}**")

            st.info(f"Recommended workflow: **{action}**")

        st.markdown("### Risk Meter")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text":"AI Risk Score"},
            gauge={
                "axis":{"range":[0,100]},
                "steps":[
                    {"range":[0,40],"color":"#174b36"},
                    {"range":[40,70],"color":"#59451d"},
                    {"range":[70,100],"color":"#5c1e2b"}
                ],
                "threshold":{"line":{"color":"white","width":4},"thickness":.8,"value":risk_score}
            }
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#eaf2ff",
            height=320,
            margin=dict(l=20,r=20,t=50,b=10)
        )
        st.plotly_chart(gauge, use_container_width=True)

        st.warning("Important: this is an academic demonstration using synthetic data. It must not be used as an actual automated lending, rejection, or fraud accusation system.")

elif page == "Model Performance":
    st.markdown("### 🧪 Model Performance")
    st.caption("Evaluation is based on the generated synthetic demonstration dataset.")

    cols = st.columns(5)
    vals = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"])
    ]
    for col,(name,val) in zip(cols, vals):
        col.metric(name, f"{val:.3f}")

    cm = metrics["confusion"]
    fig = px.imshow(
        cm, text_auto=True, x=["Predicted Legit","Predicted Fraud"],
        y=["Actual Legit","Actual Fraud"], aspect="auto",
        title="Confusion Matrix"
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce9f8"
    )
    st.plotly_chart(fig, use_container_width=True)

    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
    fig2 = px.bar(
        imp, orientation="h",
        labels={"value":"Importance","index":"Feature"},
        title="Top Model Features"
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dce9f8"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "About":
    st.markdown("### About the Project")
    st.markdown("""
    **Anti-Fraud Model for Internet Loans Using Machine Learning** is an academic
    decision-support prototype for identifying suspicious patterns in online loan
    applications.

    **Architecture**
    - Random Forest: supervised fraud-risk classification
    - Isolation Forest: anomaly detection
    - Feature engineering: debt-to-income and loan-to-income signals
    - Explainability layer: human-readable contributing factors
    - Streamlit: interactive application and dashboard

    **Important limitation**

    The bundled model is trained on synthetically generated data so the application
    can run immediately without exposing real customer information. A production
    system would require a legally obtained, representative, quality-controlled
    dataset, rigorous validation, fairness testing, security controls, model
    governance, and human review.
    """)
    st.markdown("### Suggested Production Extensions")
    st.markdown("""
    1. Replace synthetic data with an approved real-world or benchmark dataset.
    2. Add cross-validation and threshold tuning.
    3. Add SHAP explanations.
    4. Add role-based authentication and audit logs.
    5. Add model drift monitoring.
    6. Add privacy controls and data minimization.
    7. Validate the system separately for fraud detection and credit-risk decisions.
    """)

st.markdown("---")
st.caption("🛡️ Anti-Fraud AI • Academic prototype • Machine-learning assisted risk screening")
