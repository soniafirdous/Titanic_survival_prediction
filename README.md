# 🚢 Titanic Survival Prediction Project

🔎 Project Overview

This project predicts the survival of passengers on the Titanic using machine learning. 
By analyzing passenger data such as gender, age, class, fare, and family details, multiple models are trained and compared to identify the best-performing approach. 
Feature engineering is applied to improve prediction accuracy.

📌 Dataset

Training Set (train.csv) – Includes passenger details with survival labels. Used for training and evaluation.

Test Set (test.csv) – Includes passenger details without survival labels. Used for generating predictions on unseen data.

🔷 Key Features

Passenger demographics: Age, Sex

Ticket information: Pclass, Fare, Embarked

Family-related features: SibSp, Parch, FamilySize, Family

🖊 Purpose

Predict passenger survival using machine learning.

Identify key features that influence survival chances.

Compare different models (baseline and advanced) to select the best.

Generate predictions in Kaggle submission format.

✂ Feature Engineering

FamilySize: SibSp + Parch + 1

Family: 0 if FamilySize == 1 else 1 (alone vs with family)

Binned Family Size: Grouped into Small, Medium, Large for analysis.

Outlier handling and missing value imputation applied where needed.

🛢 Models Used

* Logistic Regression

* Decision Tree

* Random Forest

* Gradient Boosting

* XGBoost

* LightGBM

* Support Vector Classifier (SVC)

* K-Nearest Neighbors (KNN)

👉 Pipelines were used for preprocessing (scaling, encoding) and model training to ensure consistent and clean workflows.

🔓 Usage

Train models: Load train.csv, preprocess, and fit pipelines.

Tune hyperparameters: GridSearchCV applied to optimize models (e.g., C, depth, learning rate).

Evaluate models: Compare performance using cross-validation and test predictions.

Save models: Export best pipeline using pickle or joblib.

Generate submission:

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})
submission.to_csv("submission.csv", index=False)

🛠 Evaluation

Metrics used: Accuracy, Precision, Recall, F1-score

Visualization: Confusion matrices, classification reports, boxplots, and distributions to analyze data and results.

📊 Model Comparison (Summary)

Best Performer: Gradient Boosting (highest accuracy & F1-score).

Strong Competitors: Random Forest, XGBoost, and Decision Tree.

Baseline: Logistic Regression gave solid results with good interpretability.

Weaker Models: KNN and SVC underperformed compared to ensembles.

📍 Conclusion

Ensemble models like Gradient Boosting and XGBoost deliver the most reliable results for Titanic survival prediction.

Feature engineering (FamilySize, Family) and handling missing values/outliers significantly improve model accuracy.

Logistic Regression remains a simple but effective baseline, while advanced boosting methods capture complex interactions better.

📌 Requirements

Python 3.x

Pandas, NumPy

Scikit-learn

LightGBM, XGBoost

Matplotlib, Seaborn

👩‍💻 Author: Sonia Firdous
