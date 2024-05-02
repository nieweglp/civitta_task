1. Setup environment

First of all to run all codes, we have to create Python virtual environment.
All project was written in Python 3.10
Clone repo.
Create virtual venv using:

`virtualenv venv`

Then acticate venv using(Linux and MacOS):

`source venv/bin/activate`

In Windows run batch file from cmd:

`venv\Scripts\activate`

Then install all packages by:

`pip install -r requirements.txt`

After installation we can run jupyter notebook or jupyter-lab from terminal.


2. Dataset Exploratory Data Analysis.

Every file loan_information, client_information and loan_outcome_information contains the same number of samples. I join all files by client_id key


client_type joins early and regular as a 0 and default as 1 - one versus all strategy





Correlations
Positive significant correlations:
- vehicle_initial_assessment_value with loan_initial_amount 0.75: It means that

Negative significant correlations:
- loan_initial_term with ratio -0.48: 


We have a null values in columns: ratio(2581 samples), region(1090 samples) and age(22 samples). 
At this point we have few options how to handle missing data:
- stay with missing data and choose models which are resistant for missing data like: Decision tree, Random Forest or Ensemble methods.
- drop rows with missing values, but we loose samples to train model
- imput constants like mean, median or other values
- imput missing data using ML model e.g. Decision tree or ensemble methods.

I have checked all above possibilities and the best model statistics I had, when I used strategy to impute mean values for numerical features and most frequent for categorical variables. Surprisingly MICE algorithm (LightGBM imputer implementation) returns worse model results than constant imputation strategy.

After 
I choose MICE algorithm, which automatically fill empty values for a 

Dataset has imbalanced target variable(client_type), so we cannot look only at accuracy, because is not representative. It's better to look at the F1-score, which combines Precision and Recall.

We should balance training dataset and we can do the same on test set(but it's not necessairy).

2. ML model.
Column client_type is our target variable, which contains 3 labels: default, regular and early. So at the first stage we have mulclass classification.

We know from EDA that distributions of ratio for default and early are similar and within scope 0 to 1. Best pair, where we can find differences in distribution is default and regular. But there when we train on pair default and regular, when new client arrive we cannot 

At this point we can take few possibilities of strategy:
- one vs one: we can create classifier to train on each pair of classes
- one vs rest: we can create model, where we take 1 as a default, and regular+early client as a 0
- stay with multiclass classification



I used RandomForestClassifier and XGBoost classifier models, because their are usually given best results comparing to e.g. Decision Trees or Logistic Regression. Their are complex models, but I didn't have any limitations about the training computing time of the model. Also we have small sample, so it wasn't big challenge to train those models in my local machine.

Before I trained a ML model, I have to prepare X features, which will describe y target. To do this I have to encode categorical features like region, branch, client_gender, loan_type using one hot encoding method. Other columns I 

3. Risk/not Risky customer profile.

4. Conclusion:
We can dig in a data for more features, which will better describe each group of clients.


NOTES:

or
multiclassification with flatten
client_type its our target variable - Y
the rest columns(after feature engineering) will be - X

try some simple stats: chi square
balance dataset because default is minority group: try oversampling or smoote

experiment tracking with mlflow

models: random forest and xgboost 


under sampling, over sampling, smote K nearest
