# Dual Model Employee Salary Predictor (Classification & Regression)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.1%2B-orange.svg)

This project is an advanced web application that performs two key tasks:
1.  **Classification**: Predicts whether an employee's salary is `>50K` or `<=50K`.
2.  **Regression**: Predicts an *estimated numerical salary* for the employee.

The project stands out by creatively handling data limitations (imputing salary for the regression task) and integrating a suite of advanced machine learning and explainable AI techniques into a polished, interactive interface.

## 🚀 Key Features

- **Dual ML Models**: Simultaneously trains and serves a **Stacking Classifier** for income bracket prediction and a **Stacking Regressor** for estimating the actual salary value.
- **Creative Feature Engineering**: Implements a rule-based imputation method to create a numerical `estimated_salary` target variable from the original categorical data, enabling the regression task.
- **Dynamic Prediction Gauge**: An interactive Plotly gauge that visually represents the model's confidence in the `>50K` classification.
- **Explainable AI (XAI)**: Utilizes `LIME` to generate a plot and a table explaining *why* a prediction was made for a specific individual.
- **Styled Visualizations**: All plots are styled with color gradients and clear labels for a professional look and feel.
- **Interactive Web Interface**: A user-friendly application built with Streamlit, featuring tabs to clearly separate classification and regression insights.

## 🧠 How It Works

The application's core logic is built on a dual-model system:

1.  **Salary Imputation**: Since the original dataset only contains salary brackets (`>50K` / `<=50K`), a new numerical feature, `estimated_salary`, is engineered. This is done by assigning a base salary to each bracket and then applying premiums based on factors like education and occupation, allowing a regression model to be trained.
2.  **Feature Selection**: The `fnlwgt` (final weight) column, a statistical measure not relevant for individual prediction, has been removed to improve model focus and performance.
3.  **Stacked Ensemble Training**: Two separate stacked models are trained—one for classification and one for regression. Each ensemble uses Random Forest and Gradient Boosting as base estimators and a linear model as the final meta-learner, ensuring high performance.
4.  **Prediction & Explanation**: When a user inputs data, it is preprocessed and fed to both models. The app displays the classification result, the numerical salary estimate, and a LIME explanation detailing the factors that influenced the classification outcome.

## 🛠️ Technology Stack

- **Backend**: Python, Scikit-learn, Pandas
- **Frontend / Web App**: Streamlit
- **Data Visualization**: Matplotlib, Plotly
- **Explainable AI**: LIME
- **Deployment**: Ngrok (for local tunneling)

## ⚙️ Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Models (First-Time Setup)**: Before running the app for the first time, you need to train both the classifier and the regressor. Run the following command in your terminal:
    ```bash
    python salary_app_dual.py train
    ```
    This will create three files: `stacked_classifier_model_v2.pkl`, `stacked_regressor_model_v2.pkl`, and `preprocessor_dual_v2.pkl`.

5.  **Run the Application**: Use the helper script to launch the app.
    ```bash
    python run_ngrok.py
    ```
    You will be prompted to choose between a **Local Only** launch or a **Public Ngrok** launch. If you choose ngrok, you may be asked for your authtoken.

## ⚠️ Known Limitations & Disclaimer

- The **Predicted Estimated Salary** is derived from a regression model trained on *imputed* data, as the original dataset does not contain exact salary figures. The imputation logic is a rule-based estimation and not a reflection of true salaries.
- As a result, there may be instances where the **Predicted Salary Bracket** (e.g., `<=50K`) and the **Predicted Estimated Salary** (e.g., `$54,000`) appear to contradict.
- This is an expected outcome, as the two models learn different patterns from the data. The classifier is generally more reliable for the income bracket, while the regressor provides a ballpark numerical estimate. The application will display a warning when such a contradiction occurs.
- **Global vs. Local Importance**: The "Overall Feature Importance" plot shows which features are most important *on average* across all data. The "LIME Explanation" plot shows which features were most important for *one specific prediction*. It is normal and expected for these two plots to highlight different features.

## 🔮 Future Improvements

- **Hyperparameter Tuning**: Implement GridSearchCV or RandomizedSearchCV to find the optimal parameters for the base models and the meta-learner to further improve accuracy.
- **Advanced Imputation**: Explore more sophisticated methods for salary imputation, potentially using a preliminary regression model to generate the estimates.
- **Cloud Deployment**: Deploy the application on a permanent cloud service like Streamlit Community Cloud or Heroku for 24/7 availability.
