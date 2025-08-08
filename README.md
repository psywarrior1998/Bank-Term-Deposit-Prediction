# Bank Term Deposit Subscription Prediction

![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S5E8-blue)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 1. Project Overview

This project presents a complete, end-to-end machine learning solution for the **[Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)** Kaggle competition. The primary objective is to predict whether a bank client will subscribe to a term deposit based on their marketing campaign data.

The solution employs a robust pipeline that includes data preprocessing, hyperparameter tuning of a diverse suite of models, and the creation of a high-performing soft-voting ensemble. The entire workflow is documented and reproducible, with all key assets, such as trained models and result visualizations, saved for future use.

---

## 2. The Dataset

The data is sourced from the Kaggle Playground Series (Season 5, Episode 8) and contains a mix of numerical and categorical features describing bank client data and their interactions with a marketing campaign. The task is a binary classification problem evaluated on the **ROC AUC** metric.

---

## 3. Project Workflow

This project follows a structured and rigorous machine learning workflow designed for robustness and high performance.

1.  **Data Preprocessing**:
    * **Scaling**: Numerical features are standardized using `StandardScaler` to prevent features with large value ranges from dominating the model.
    * **Encoding**: Categorical features are converted into a machine-readable format using `OneHotEncoder`.
    * All preprocessing steps are encapsulated in a `ColumnTransformer` and `Pipeline` to prevent data leakage and ensure consistency.

2.  **Hyperparameter Tuning**:
    * A diverse suite of seven powerful classifiers was selected to cover different algorithmic families (linear models, tree-based ensembles, support vector machines, etc.).
    * `GridSearchCV` was employed with 5-fold cross-validation to systematically search for the optimal hyperparameters for each model, maximizing the `roc_auc` score.

3.  **Elite Ensemble Modeling**:
    * The top 4 performing models from the tuning phase were strategically selected to form a final ensemble.
    * A `VotingClassifier` with `voting='soft'` was used to average the predicted probabilities from these elite models, leveraging the confidence of each model to produce a more accurate and robust final prediction.

4.  **Comprehensive Evaluation**:
    * The final ensemble model was rigorously evaluated on a held-out validation set using a full suite of metrics, including a detailed **Classification Report**, **Confusion Matrix**, **ROC Curve**, and **Precision-Recall Curve**.

5.  **Model Persistence & Feature Analysis**:
    * All tuned individual models and the final ensemble model were serialized and saved to disk using `joblib`.
    * Key visualizations and feature importance plots were generated and saved to provide insights into the model's decision-making process.

---

## 4. Repository Structure

```

.
├── saved\_graphs/
│   ├── model\_performance\_comparison.png
│   ├── confusion\_matrix.png
│   ├── performance\_curves.png
│   └── feature\_importance.png
│
├── saved\_models/
│   ├── LogisticRegression\_best\_model.pkl
│   ├── RandomForest\_best\_model.pkl
│   ├── ... (all other tuned models) ...
│   └── elite\_ensemble\_model.pkl
│
├── .gitignore
├── README.md
├── requirements.txt
└── bank_subscription_prediction.ipynb

````

---

## 5. Key Results & Visualizations

The visualizations below summarize the performance of the models and the final ensemble.

#### Model Performance Comparison
This chart ranks the individual tuned models based on their cross-validated ROC AUC scores, justifying the selection for the final ensemble.
<br>
<img src="saved_graphs/model_performance_comparison.png" alt="Model Performance" width="600"/>

#### Ensemble Performance
The confusion matrix and performance curves demonstrate the high predictive power of the final elite ensemble on the validation set.
<br>
<img src="saved_graphs/confusion_matrix.png" alt="Confusion Matrix" width="400"/> <img src="saved_graphs/performance_curves.png" alt="Performance Curves" width="700"/>

#### Feature Importance
This plot reveals the key drivers behind the model's predictions, with call duration (`duration`) being the most influential feature.
<br>
<img src="saved_graphs/feature_importance.png" alt="Feature Importance" width="500"/>

---

## 6. How to Run

To replicate this project and its results, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Bank-Term-Deposit-Prediction.git](https://github.com/YourUsername/Bank-Term-Deposit-Prediction.git)
    cd Bank-Term-Deposit-Prediction
    ```

2.  **Install dependencies:**
    It is recommended to create a virtual environment first.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebook/script:**
    Open and run the main Jupyter Notebook (`bank_subscription_prediction.ipynb`) or Python script. This will execute the entire workflow, from data loading to saving the final submission file (`submission.csv`).

---

## 7. Technologies Used

* **Core Libraries**: `Python`, `Pandas`, `NumPy`
* **Machine Learning**: `Scikit-Learn`, `XGBoost`, `LightGBM`
* **Visualization**: `Matplotlib`, `Seaborn`
* **Model Persistence**: `Joblib`

---

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for details.
