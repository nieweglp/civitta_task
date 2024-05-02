1. Setup environment

First of all to run all codes, we have to create Python virtual environment.
All project was written in Python 3.10
Clone repo using `git clone https://github.com/nieweglp/civitta_task.git`
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
- vehicle_initial_assessment_value with loan_initial_amount 0.75: loan initial amount is increasing, when vehicle price is increasing. Clients usually need higher loan for expensive cars.
- monthly_interest_rate with annual_percentage_rate 0.64: it's logical, but changing level of interest_rate might noise this statistic
- income with loan_initial_amount 0.32: here is not strong correleation, but usually when income is high you have higher credit scoring
- vehicle_production_year with loan_initial_amount 0.45: when car is new we need higher loan 
- ratio with loan_initial_term: usually higher ratio when 
- loan_initial_amount with loan_initial_term 0.31: usually when loan is higher clients needs longer term to pay off credit

Negative significant correlations:
- loan_initial_term with ratio -0.48: loan initial term is decreasing for higher ratio, from other visualization know that higher ratio has regular 


We have a null values in columns: ratio(2581 samples), region(1090 samples) and age(22 samples). 
At this point we have few options how to handle missing data:
- stay with missing data and choose models which are resistant for missing data like: Decision tree, Random Forest or Ensemble methods.
- drop rows with missing values, but we loose samples to train model
- imput constants like mean, median or other values
- imput missing data using ML model e.g. Decision tree or ensemble methods.

I have checked all above possibilities and the best model statistics I had, when I used strategy to impute mean values for numerical features and most frequent for categorical variables. Surprisingly MICE algorithm (LightGBM imputer implementation) returns worse model results than constant imputation strategy.

After I choose MICE algorithm, which automatically fill empty values based on the other columns, I saw that this method performed worse than mean imputer.

Dataset has imbalanced target variable(client_type), so we cannot look only at accuracy, because is not representative. It's better to look at the F1-score, which combines Precision and Recall.

We should balance training dataset and we can do the same on test set(but it's not necessairy).

2. ML model.
Column client_type is our target variable, which contains 3 labels: default, regular and early. So at the first stage we have mulclass classification.

We know from EDA that distributions of ratio for default and early are similar and within scope 0 to 1. Best pair, where we can find differences in distribution is default and regular. But there when we train on pair default and regular, when new client arrive we cannot 

At this point we can take few possibilities of strategy:
- one vs one: we can create classifier to train on each pair of classes
- one vs rest: we can create model, where we take 1 as a default, and regular+early client as a 0
- stay with multiclass classification

I choose to train model, where 1 will be default clients and 0 labaled regular&early clients. Data shows different perspective, but from business point of view early and regular clients are in the same group(cluster).

I used RandomForestClassifier and XGBoost classifier models, because their are usually given best results comparing to e.g. Decision Trees or Logistic Regression. Their are complex models, but I didn't have any limitations about the training computing time of the model. Also we have small sample, so it wasn't big challenge to train those models in my local machine.

Before I trained a ML model, I have to prepare X features, which will describe y(client_type) target variable. To do this I have to encode categorical features like region, branch, client_gender, loan_type using one hot encoding method.

3. Risk/not Risky client profile.

Usually lower risk client is a client, who has higher income.
Higher risk of default is for clients with higher interest rates.

But we have to remember that in the data we have a lot of noises(null value, outliers), which disturb generalization.

Based on my week experiments on those 2 models I can see that Random Forest Classifier performed better. Impact for this might have difference in balancing training dataset(under and over sampling techniques). 

Ratio has huge impact into default prediction for both models.. Another important features are: monthly_interest_rate, anual_percentage_rate, loan_initial_term, loan_initial_amount. As we saw Random Forest Classifer takes similar features like XGBoost, but in a different order with difference importance.

High Shapley value for feature increasing probability of pay loan on time. From the other hand lower Shapley values decrease probability of loan default.


4. Conclusion:

The biggest challenge in this dataset is that early client looks familiar to default. Distribution per each type is similiar. It's quite challenging to find diffrences when, we have the same distribution in the same range. Ratio column clearly shows this effect. 

Increase dataset samples to improve model accuracy(loyalty program to collect data give historical behaviour knowledge).
Optimize/autoscaling model parameters to enhance accuracy.
Feature engineering - gathering new features or combining existing ones

Reducing or fullify empty values in features: ratio, age and branch.

Model should reduce priority False Negative cases(model mark default client as a good) and then False Positive cases(model mark good clients as a default).
We should know cost($) of False Positive and False Negative for a one client.
To keep clients(regular and early) satisfaction we should give them discount benefits or set loyalty program(it will collect data to e.g. data warehouse).
Know better default clients(reason of delays).

“Brain storming” with business owner about default problem to get more business insights e.g. How differ regular and early client?

