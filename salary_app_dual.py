import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from lime.lime_tabular import LimeTabularExplainer
import os
import warnings

# Suppress the harmless UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Constants and Configuration ---
DATA_PATH = 'adult 3.csv'
CLASSIFIER_MODEL_PATH = "stacked_classifier_model_v3.pkl"
REGRESSOR_MODEL_PATH = "stacked_regressor_model_v3.pkl"
PREPROCESSOR_PATH = "preprocessor_dual_v3.pkl"

# --- Advanced Feature Engineering: Salary Imputation ---
def advanced_impute_salary(df):
    """
    Creates a more realistic estimated numerical salary using a preliminary regression model.
    """
    df_imputed = df.copy()
    df_imputed.replace('?', np.nan, inplace=True)
    df_imputed.dropna(inplace=True)

    # Quick preprocessing for the preliminary model
    df_temp = df_imputed.copy()
    for col in ['education', 'fnlwgt']:
        if col in df_temp.columns:
            df_temp.drop(col, axis=1, inplace=True)
    
    categorical_cols = df_temp.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')

    for col in categorical_cols:
        le = LabelEncoder()
        df_temp[col] = le.fit_transform(df_temp[col])
    
    # Create a simple numerical target for the preliminary model
    df_temp['income_numeric'] = df_temp['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    X_prelim = df_temp.drop(['income', 'income_numeric'], axis=1)
    y_prelim = df_temp['income_numeric']
    
    # Train a simple, fast model to get a continuous prediction
    prelim_model = Ridge()
    prelim_model.fit(X_prelim, y_prelim)
    continuous_prediction = prelim_model.predict(X_prelim)
    
    # Scale the continuous prediction to a realistic salary range
    min_salary = 25000
    max_salary = 150000
    scaled_salary = min_salary + (continuous_prediction - continuous_prediction.min()) / (continuous_prediction.max() - continuous_prediction.min()) * (max_salary - min_salary)
    
    df_imputed['estimated_salary'] = scaled_salary + np.random.uniform(-2000, 2000)
    return df_imputed

# --- Data Preprocessing ---
def preprocess_data(df):
    """
    Cleans, encodes, and removes the 'fnlwgt' feature.
    """
    df_proc = df.copy()
    df_proc.replace('?', np.nan, inplace=True)
    df_proc.dropna(inplace=True)

    # Drop redundant or irrelevant columns
    for col in ['education', 'fnlwgt']:
        if col in df_proc.columns:
            df_proc.drop(col, axis=1, inplace=True)

    categorical_cols = df_proc.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        label_encoders[col] = le

    le_income = LabelEncoder()
    df_proc['income_class'] = le_income.fit_transform(df_proc['income'])
    label_encoders['income'] = le_income
    
    df_proc.drop('income', axis=1, inplace=True)
    return df_proc, label_encoders

# --- Model Training ---
def train_models():
    """
    Loads data, preprocesses it, performs hyperparameter tuning, 
    trains both a classifier and a regressor, and saves them.
    """
    print("--- Starting Dual Model Training ---")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset '{DATA_PATH}' not found.")
        return

    print("Loading, performing advanced salary imputation, and preprocessing data...")
    df_raw = pd.read_csv(DATA_PATH)
    df_imputed = advanced_impute_salary(df_raw)
    df_processed, label_encoders = preprocess_data(df_imputed)
    print("Data ready for training!")
    print("-" * 35)

    # --- Classifier Training with Hyperparameter Tuning ---
    print("\n--- Training and Tuning Classifier ---")
    X_class = df_processed.drop(['income_class', 'estimated_salary'], axis=1)
    y_class = df_processed['income_class']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)
    
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    # Define parameter grids for RandomizedSearchCV
    param_grid_rf_c = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
    param_grid_gb_c = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}

    # Create RandomizedSearchCV objects with more iterations
    rf_c_tuned = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_rf_c, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    gb_c_tuned = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb_c, n_iter=10, cv=3, random_state=42, n_jobs=-1)

    estimators_c = [('rf', rf_c_tuned), ('gb', gb_c_tuned)]
    classifier = StackingClassifier(estimators=estimators_c, final_estimator=LogisticRegression(), cv=5)
    
    print("Tuning hyperparameters... (This may take several minutes)")
    classifier.fit(X_train_c_scaled, y_train_c)
    
    y_pred_c = classifier.predict(X_test_c_scaled)
    print("\n--- Classifier Performance ---")
    print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")
    report = classification_report(y_test_c, y_pred_c, target_names=label_encoders['income'].classes_)
    print("Classification Report:\n", report)

    # --- Regressor Training with Hyperparameter Tuning ---
    print("\n--- Training and Tuning Regressor ---")
    X_reg = df_processed.drop(['income_class', 'estimated_salary'], axis=1)
    y_reg = df_processed['estimated_salary']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    param_grid_rf_r = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
    param_grid_gb_r = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}
    
    rf_r_tuned = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid_rf_r, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    gb_r_tuned = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb_r, n_iter=10, cv=3, random_state=42, n_jobs=-1)

    estimators_r = [('rf', rf_r_tuned), ('gb', gb_r_tuned)]
    regressor = StackingRegressor(estimators=estimators_r, final_estimator=LinearRegression(), cv=5)
    
    print("Tuning hyperparameters... (This may take several minutes)")
    regressor.fit(X_train_r_scaled, y_train_r)
    
    y_pred_r = regressor.predict(X_test_r_scaled)
    print("\n--- Regressor Performance ---")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test_r, y_pred_r)):,.2f}")

    # --- Save Models and Preprocessor ---
    print("\n--- Saving Models and Preprocessor ---")
    preprocessor = {
        'label_encoders': label_encoders, 
        'scaler_c': scaler_c, 
        'scaler_r': scaler_r, 
        'feature_columns': list(X_class.columns),
        'mean_salary': y_train_r.mean()
    }
    joblib.dump(classifier, CLASSIFIER_MODEL_PATH)
    joblib.dump(regressor, REGRESSOR_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("✅ All models and preprocessor saved successfully!")
    print("-" * 35)

# --- Streamlit App UI ---
def run_app():
    st.set_page_config(page_title="Dual Salary Predictor", page_icon="💼", layout="wide")
    st.title("👨‍💼 Dual Model Salary Predictor (Classification & Regression)")
    
    if not all(os.path.exists(p) for p in [CLASSIFIER_MODEL_PATH, REGRESSOR_MODEL_PATH, PREPROCESSOR_PATH]):
        st.warning("Models not found. Please train the models first by running `python salary_app_dual.py train` in your terminal.")
        return

    classifier = joblib.load(CLASSIFIER_MODEL_PATH)
    regressor = joblib.load(REGRESSOR_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    st.sidebar.header("Employee Details")

    # Input fields
    age = st.sidebar.slider("Age", 17, 90, 35)
    workclass = st.sidebar.selectbox("Work Class", preprocessor['label_encoders']['workclass'].classes_)
    educational_num = st.sidebar.slider("Education Level (Num)", 1, 16, 10)
    marital_status = st.sidebar.selectbox("Marital Status", preprocessor['label_encoders']['marital-status'].classes_)
    occupation = st.sidebar.selectbox("Occupation", preprocessor['label_encoders']['occupation'].classes_)
    relationship = st.sidebar.selectbox("Relationship", preprocessor['label_encoders']['relationship'].classes_)
    race = st.sidebar.selectbox("Race", preprocessor['label_encoders']['race'].classes_)
    gender = st.sidebar.selectbox("Gender", preprocessor['label_encoders']['gender'].classes_)
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
    native_country = st.sidebar.selectbox("Native Country", preprocessor['label_encoders']['native-country'].classes_)

    input_data = {'age': age, 'workclass': workclass, 'educational-num': educational_num, 'marital-status': marital_status, 'occupation': occupation, 'relationship': relationship, 'race': race, 'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}
    input_df = pd.DataFrame([input_data])
    
    st.subheader("Your Input:")
    st.dataframe(input_df)

    processed_input = input_df.copy()
    for col, le in preprocessor['label_encoders'].items():
        if col in processed_input.columns and col != 'income':
            known_labels = le.classes_
            processed_input[col] = processed_input[col].apply(lambda x: x if x in known_labels else known_labels[0])
            processed_input[col] = le.transform(processed_input[col])

    if st.button("Predict Salary", key='predict'):
        scaled_input_c = preprocessor['scaler_c'].transform(processed_input)
        prediction_c = classifier.predict(scaled_input_c)
        prediction_proba_c = classifier.predict_proba(scaled_input_c)
        predicted_class = preprocessor['label_encoders']['income'].inverse_transform(prediction_c)[0]

        scaled_input_r = preprocessor['scaler_r'].transform(processed_input)
        prediction_r = regressor.predict(scaled_input_r)
        
        st.subheader("📈 Prediction Overview")
        
        if predicted_class == '<=50K' and prediction_r[0] > 50000:
            st.warning("""
            ⚠️ **Prediction Insight:** The models show a conflicting result.
            - **Classifier Prediction:** The model is confident the individual belongs to the **<=50K** salary bracket.
            - **Regressor Prediction:** The model estimates a salary **above $50,000**.
            
            This is an expected outcome due to the models' different natures. The classifier focuses on the dividing line between income groups, while the regressor predicts a specific value based on imputed data that may not perfectly reflect real-world salaries. **The classification result is generally more reliable for determining the income bracket.**
            """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Salary Bracket", value=predicted_class)
        with col2:
            st.metric(label="Predicted Estimated Salary", value=f"${prediction_r[0]:,.2f}")

        # --- TABS FOR DETAILED INSIGHTS ---
        tab_c, tab_r = st.tabs(["**Classifier Insights**", "**Regressor Insights**"])

        with tab_c:
            st.subheader("Classifier Model Analysis")
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                st.markdown("##### Probability Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=prediction_proba_c[0][1] * 100,
                    title={'text': "Probability of Earning >50K (%)"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#1f77b4"},
                           'steps': [{'range': [0, 50], 'color': '#FFB6C1'}, {'range': [50, 100], 'color': '#90EE90'}]},
                    number={'suffix': '%'}
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with c_col2:
                st.markdown("##### Overall Feature Importance")
                rf_model_c = classifier.named_estimators_['rf'].best_estimator_
                importances_c = rf_model_c.feature_importances_
                feature_importance_c = pd.DataFrame({'feature': preprocessor['feature_columns'], 'importance': importances_c}).sort_values('importance', ascending=False)
                
                fig_imp_c, ax_imp_c = plt.subplots(figsize=(6, 4))
                ax_imp_c.barh(feature_importance_c['feature'], feature_importance_c['importance'], color='skyblue', edgecolor='black')
                ax_imp_c.invert_yaxis()
                st.pyplot(fig_imp_c)

            st.markdown("---")
            st.markdown("##### LIME Explanation (Why this prediction?)")
            with st.spinner("Generating Classifier LIME explanation..."):
                df_raw_lime = pd.read_csv(DATA_PATH)
                df_imputed_lime = advanced_impute_salary(df_raw_lime)
                df_processed_lime, _ = preprocess_data(df_imputed_lime)
                X_lime = df_processed_lime.drop(['income_class', 'estimated_salary'], axis=1)
                
                lime_explainer_c = LimeTabularExplainer(
                    training_data=preprocessor['scaler_c'].transform(X_lime.values),
                    feature_names=preprocessor['feature_columns'],
                    class_names=['<=50K', '>50K'], mode='classification'
                )
                exp_c = lime_explainer_c.explain_instance(scaled_input_c[0], classifier.predict_proba, num_features=10, labels=(1,))
                
                fig_lime_c, ax_lime_c = plt.subplots(figsize=(10, 4))
                lime_vals_c = exp_c.as_list(label=1)
                labels_c = [lv[0] for lv in lime_vals_c]
                vals_c = [lv[1] for lv in lime_vals_c]
                colors_c = ['#90EE90' if v > 0 else '#FFB6C1' for v in vals_c]
                ax_lime_c.barh(labels_c, vals_c, color=colors_c, edgecolor='black')
                ax_lime_c.set_xlabel('Contribution to >50K Probability')
                st.pyplot(fig_lime_c)

                st.markdown("#### Explanation Details")
                impact_data_c = []
                for label, val in lime_vals_c:
                    clean_feat = label.split(' ')[0]
                    if clean_feat not in input_data:
                        clean_feat = label.split('=')[0].strip()
                    original_val = str(input_data.get(clean_feat, 'N/A'))
                    impact_data_c.append({
                        'Feature Rule': label,
                        'Your Value': original_val,
                        'Impact on >50K': f'{val:.3f}',
                        'Effect': 'Increases Likelihood' if val > 0 else 'Decreases Likelihood'
                    })
                st.dataframe(pd.DataFrame(impact_data_c))
            
            st.info("""
                **Understanding the Plots:**
                - **LIME Explanation (Local):** Shows the top features influencing *this specific prediction*. It answers: "Why did the model decide this for this person?"
                - **Overall Importance (Global):** Shows the features the model found most important *on average* across the entire dataset. It answers: "What features does the model generally find most predictive?"
                It is normal for these two plots to highlight different features.
            """)

        with tab_r:
            st.subheader("Regressor Model Analysis")
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("##### Salary Indicator")
                fig_indicator = go.Figure(go.Indicator(
                    mode = "number+delta",
                    value = prediction_r[0],
                    title = {"text": "Predicted Salary vs. Average"},
                    delta = {'reference': preprocessor['mean_salary'], 'relative': False, 'valueformat': ',.0f'},
                    number = {'prefix': "$", 'valueformat': ',.0f'}
                ))
                fig_indicator.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_indicator, use_container_width=True)
                st.caption("The green value indicates the prediction is higher than the dataset's average salary. A red value would indicate it's lower.")

            with r_col2:
                st.markdown("##### Overall Feature Importance")
                rf_model_r = regressor.named_estimators_['rf'].best_estimator_
                importances_r = rf_model_r.feature_importances_
                feature_importance_r = pd.DataFrame({'feature': preprocessor['feature_columns'], 'importance': importances_r}).sort_values('importance', ascending=False)
                
                fig_imp_r, ax_imp_r = plt.subplots(figsize=(6, 4))
                ax_imp_r.barh(feature_importance_r['feature'], feature_importance_r['importance'], color='salmon', edgecolor='black')
                ax_imp_r.invert_yaxis()
                st.pyplot(fig_imp_r)

            st.markdown("---")
            st.markdown("##### LIME Explanation (Why this salary estimate?)")
            with st.spinner("Generating Regressor LIME explanation..."):
                lime_explainer_r = LimeTabularExplainer(
                    training_data=preprocessor['scaler_r'].transform(X_lime.values),
                    feature_names=preprocessor['feature_columns'],
                    mode='regression'
                )
                exp_r = lime_explainer_r.explain_instance(scaled_input_r[0], regressor.predict, num_features=10)
                
                fig_lime_r, ax_lime_r = plt.subplots(figsize=(10, 4))
                lime_vals_r = exp_r.as_list()
                labels_r = [lv[0] for lv in lime_vals_r]
                vals_r = [lv[1] for lv in lime_vals_r]
                colors_r = ['#90EE90' if v > 0 else '#FFB6C1' for v in vals_r]
                ax_lime_r.barh(labels_r, vals_r, color=colors_r, edgecolor='black')
                ax_lime_r.set_xlabel('Contribution to Salary ($)')
                st.pyplot(fig_lime_r)

                st.markdown("#### Explanation Details")
                impact_data_r = []
                for label, val in lime_vals_r:
                    clean_feat = label.split(' ')[0]
                    if clean_feat not in input_data:
                        clean_feat = label.split('=')[0].strip()
                    original_val = str(input_data.get(clean_feat, 'N/A'))
                    impact_data_r.append({
                        'Feature Rule': label,
                        'Your Value': original_val,
                        'Impact on Salary': f'${val:,.2f}',
                        'Effect': 'Increases Salary' if val > 0 else 'Decreases Salary'
                    })
                st.dataframe(pd.DataFrame(impact_data_r))
            
            st.info("""
                **Understanding the Plots:**
                - **LIME Explanation (Local):** Shows how the features of this specific person pushed the salary estimate up or down.
                - **Overall Importance (Global):** Shows which features the model generally uses to estimate salaries across all people.
            """)

# --- Main Execution Logic ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_models()
    else:
        run_app()
