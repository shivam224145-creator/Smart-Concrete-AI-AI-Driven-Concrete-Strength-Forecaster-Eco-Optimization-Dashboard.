# 🏗️ SmartConcrete AI: Core Concrete Optimization Suite

An integrated, end-to-end Machine Learning and Deep Learning system designed to predict concrete compressive strength, evaluate algorithmic performance, analyze structural hardening timelines, and simulate sustainable green concrete recipes.

### Web app link - https://2vpmbni3hry2cwkcx7y9va.streamlit.app/

---

## 🌟 Project Overview
Concrete is the backbone of modern infrastructure, but traditional testing methods take **28 days** to determine material strength. Additionally, cement production accounts for roughly 8% of global CO2 emissions. 

**SmartConcrete AI** solves both challenges simultaneously:
1. **Predictive Analytics:** Uses a Deep Neural Network (MLP) to instantly map 8 complex chemical components to continuous strength outputs (MPa).
2. **Sustainability Engineering:** Simulates and optimizes alternative "Green Concrete" formulations by substituting cement with industrial by-products (Fly Ash & Slag) without sacrificing load-bearing capacity.

---

## 📦 Project Structure & Modules

The project is structured into four distinct modules integrated into a unified multi-tab application:

*   **🔮 Tab 1: Strength Predictor** - Core regression deployment module powered by a Deep Artificial Neural Network mapping 8 dynamic variables.
*   **🍃 Tab 2: EcoMix Optimizer** - A carbon mitigation engine that allows users to scale down carbon-intensive cement inputs and dynamically balance strengths with industrial recycled aggregates.
*   **📈 Tab 3: Hardening Dynamics** - An interactive time-series timeline visualizer charting progressive concrete hydration states over a 90-day milestone framework.
*   **📊 Tab 4: Benchmarking Report** - A comparative statistical evaluation grid benchmarking Linear Regression, Random Forest, and Deep Learning Multi-Layer Perceptrons.

---

## 🏗️ Technical Architecture & Pipeline

### 1. Repository Directory Structure
```text
📁 SmartConcrete_AI/
│
├── 📁 data/                  
│   ├── 📁 raw/               # Original structural tests raw CSV file
│   └── 📁 cleaned/           # Preprocessed dataset (duplicates dropped: 1005 unique rows)
│
├── 📁 notebooks/             # Jupyter Research & Exploratory Data Analysis Environment
│   └── concrete_analytics_and_modeling.ipynb
│
├── 📁 models/                # Production-ready serialization checkpoints
│   ├── concrete_nn_model.keras  # Core Deep Learning Weights Engine (70 KB)
│   └── standard_scaler.pkl   # Fitted Standardizer to avoid data leakage
│
├── 📁 app/                   
│   └── app.py                # Multi-tab responsive Streamlit dashboard deployment file
│
├── .gitignore                # Environment cache configuration safety filters (.idea/, etc.)
└── requirements.txt          # Production environment tracking dependency checklist
```

### 2. Algorithmic Evaluation Benchmarks

| Algorithmic Framework Model | Target Test R² Score | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: |
| **Linear Regression Baseline** | 0.5802 | 8.8953 MPa | 11.1913 MPa |
| **Deep Learning Neural Network (MLP)** | **0.8883** | **4.3704 MPa** | **5.7723 MPa** |
| **Random Forest Regressor Benchmark** | 0.9076 | 3.4841 MPa | 5.2496 MPa |

---

## 🚀 Execution & Deployment Instructions

### Local Local Machine Startup
1. **Navigate to the app environment directory:**
   ```bash
   cd "C:\Users\lenovo\OneDrive\Desktop\Smart Concrete AI\app"
   ```
2. **Boot up the responsive framework dashboard engine:**
   ```bash
   python -m streamlit run app.py
   ```

### Live Streamlit Cloud Deployment
This application features universal relative server directory tracking paths (`os.path.dirname`), making it fully compatible with Streamlit Cloud container hosts without path collision errors. 
1. Push your audited branch parameters cleanly onto your GitHub remote repository.
2. Sign in to your [Streamlit Community Cloud](https://streamlit.io) user hub.
3. Link your active workspace repository, configure your main branch pointer, choose `app/app.py` as the entry executable script file path, and deploy instantly!

---

## 🛠️ Environment Prerequisites & Dependencies
The production runtime requires tracking dependencies defined in your `requirements.txt`:
*   `streamlit`
*   `tensorflow` (Inbuilt Keras 3 architecture backend graph engine)
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `matplotlib`
*   `seaborn`
*   `joblib`

---

## 👤 Developer Profile
<div align="left">
    <table>
        <tr>
            <td><strong>Lead Framework Developer</strong></td>
            <td>SHIVAM SHUKLA</td>
        </tr>
        <tr>
            <td><strong>Core Engine Mastery</strong></td>
            <td>Data Science & Machine Learning Solutions Architecture</td>
        </tr>
        <tr>
            <td><strong>System Operations</strong></td>
            <td>End-to-End Deep Neural Networks Pipeline Engineering & Clean UI Dashboard Deployments</td>
        </tr>
    </table>
</div>

---
*Developed under global compliance guidelines for technical production repositories.*
