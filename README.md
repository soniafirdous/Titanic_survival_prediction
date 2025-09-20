# 🚢Titanic Survival Prediction Project

🔎Project Overview

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. By analyzing passenger data such as gender, age, class, and family information, the project builds models to classify whether a passenger survived or not. Feature engineering is applied to create additional useful features, improving model performance.

📌Dataset

Training Set (train.csv): Contains passenger information along with the survival outcome (Survived). Used to train and evaluate machine learning models.

Test Set (test.csv): Contains passenger information without survival outcomes. Used to generate predictions for unseen data.

🔷Features include:

Passenger demographics: Age, Sex

Ticket information: Pclass, Fare, Embarked

Family-related features: SibSp, Parch, FamilySize, Family

🖊Purpose

Predict passenger survival using machine learning.

Identify important features that influence survival.

Compare multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM) to select the best-performing one.

Generate predictions for unseen test data in a Kaggle-style submission format.

✂Features Engineering

FamilySize: Sum of siblings/spouses and parents/children aboard + 1

Family: 0 if alone, 1 if with family

🛢Models Used

1.Logistic Regression

2.Decision Tree

3.Random Forest

4.Gradient Boosting

5.XGBoost

6.LightGBM

🧵 Pipelines were used for preprocessing (scaling, encoding) and modeling, ensuring consistency between training and prediction.

🔓Usage

Train the model: Load train.csv and fit the pipeline.

Save the trained model: Use pickle or joblib to save the pipeline.

Predict on test set: Load test.csv, add derived features if needed, and generate predictions.

Generate submission: Create a CSV file with PassengerId and predicted Survived values.

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})
submission.to_csv("submission.csv", index=False)


🛠Evaluation

Accuracy, precision, recall, and F1-score were used to evaluate model performance on the training/validation set.

Confusion matrices and classification reports provide insight into how well the models identify survivors vs non-survivors.

📌Requirements

Python 3.x

Pandas

NumPy

Scikit-learn

LightGBM / XGBoost (optional)

Matplotlib / Seaborn (for visualization)

📍Conclusion

This project demonstrates how machine learning can predict survival outcomes on historical data.
Ensemble models like XGBoost and Gradient Boosting typically perform best, while feature engineering (e.g., family features) improves prediction accuracy.

🙋‍♀️Author: Sonia Firdous
