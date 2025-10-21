import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# --- 1. Data Loading and Preprocessing Function ---
@st.cache_data
def load_data_and_preprocess():
    """Loads the data, imputes missing values, scales features, and applies SMOTE."""
    df = pd.read_csv("water_potability.csv")

    # Impute missing values with column means
    df['ph'].fillna(df['ph'].mean(), inplace=True)
    df['Sulfate'].fillna(df['Sulfate'].mean(), inplace=True)
    df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)

    # Set features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

    return X_train, X_test, y_train, y_test, X.columns

# --- 2. Model Training and Evaluation Function ---
@st.cache_resource
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    """Trains and evaluates a Random Forest model using the best parameters."""
    
    # Your obtained best parameters
    best_params = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200}
    
    best_rf = RandomForestClassifier(random_state=42, **best_params)
    best_rf.fit(X_train, y_train)
    
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:,1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return best_rf, report, conf_matrix, y_test, y_proba

# --- 3. Streamlit Application Interface ---
def main():
    st.set_page_config(page_title="Water Potability Prediction Model Analysis", layout="wide")
    st.title("Water Potability Prediction Model Analysis (Random Forest)") 
    st.markdown("---")

    # Load and Preprocess Data
    X_train, X_test, y_train, y_test, feature_names = load_data_and_preprocess()

    # Model Training and Evaluation
    best_rf, report, conf_matrix, y_test, y_proba = train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # 1. Model Performance
    st.header("1. Model Performance") 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose().round(2)
        # Display everything except the final 'macro avg' and 'weighted avg' rows
        st.dataframe(report_df.iloc[[0, 1, 3, 4]])
        st.write(f"*Accuracy:* {report_df.loc['accuracy', 'precision']:.2f}")

    with col2:
        st.subheader("Confusion Matrix and AUC") # Confusion Matrix ও AUC
        st.write(f"*AUC (Area Under Curve):* {roc_auc_score(y_test, y_proba):.4f}")
        
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                    xticklabels=['Not Potable (0)', 'Potable (1)'], 
                    yticklabels=['Not Potable (0)', 'Potable (1)'])
        ax_cm.set_title('Confusion Matrix')
        st.pyplot(fig_cm)

    st.markdown("---")

    # 2. Feature Importances
    st.header("2. Feature Importances") 
    
    importances = best_rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
    feat_imp.plot(kind='bar', ax=ax_imp, color='teal')
    ax_imp.set_title("Feature Importances (Random Forest)")
    ax_imp.set_ylabel("Importance Score")
    ax_imp.tick_params(axis='x', rotation=45)
    st.pyplot(fig_imp)
    
    st.subheader("Order of Important Features (Value):") 
    st.dataframe(feat_imp.to_frame(name="Importance").round(4))

    st.markdown("---")

    # 3. ROC Curve
    st.header("3. ROC Curve") 
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})", color='darkorange')
    ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

if __name__ == '__main__':
    main()
    