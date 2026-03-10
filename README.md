# Credit Scoring PD (Probability of Default) Model

## 📌 Project Overview
This repository features a complete end-to-end machine learning pipeline for building a **Credit Scoring Model** to predict the **Probability of Default (PD)**. Credit scoring is a critical process in the financial industry, allowing lenders to assess the risk of a borrower failing to make required payments.

The project utilizes **Logistic Regression**, which is widely favored in credit risk for its high interpretability and the ease with which it can be converted into a standard credit scorecard.

## 🛠️ Project Workflow
The notebook is structured into clear, logical phases:

1.  **Data Pre-Processing**: 
    * Initial data ingestion and exploratory analysis.
    * Handling missing values and identifying significant data anomalies (e.g., age values of `-500`).
2.  **Feature Engineering (WoE & IV)**:
    * **Weight of Evidence (WoE) Transformation**: Categorical and numerical variables were transformed into WoE values to linearize relationships with the target variable.
    * **Univariate Gini Analysis**: Each feature's predictive power was individually assessed using Gini coefficients to select the most impactful variables and reduce model complexity.
3.  **Model Training**: 
    * Implementing a Logistic Regression classifier using `scikit-learn`.
    * Splitting data into Training and Testing sets to ensure model generalizability.
4.  **Performance Evaluation**:
    * **Gini Coefficient**: The model achieved a stable Gini of approximately **59%** on both training and test datasets.
    * **Visualization**: Generation of ROC Curves and Confusion Matrices to evaluate precision and recall.
5.  **Deployment Simulation**: 
    * Applying the trained model to new "production" data to generate real-time PD scores for potential borrowers.

## ⚠️ Methodology & Thresholds
**Note to Reviewers:** The classification thresholds and specific binning parameters (WoE) used in this project were determined based on **specific task requirements** and the provided dataset. These thresholds are designed for this demonstration and should not be taken as universal "best practices." In a real-world banking scenario, these are typically adjusted based on a financial institution's risk appetite and regulatory guidelines (e.g., Basel III/IV).

## 📊 Model Performance
- **Train Gini Score**: ~56.26%
- **Test Gini Score**: ~56.23%
- The close proximity of these scores indicates a robust model that does not suffer from significant overfitting.

## 💻 Technologies Used
- **Python 3.x**
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Visualization**: Matplotlib, Seaborn
- **Statistical Analysis**: SciPy (Stats)

## 🚀 How to Use
1. Clone this repository.
2. Ensure you have the `credit_score.csv` dataset in the project root.
3. Install the required libraries:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn scipy
