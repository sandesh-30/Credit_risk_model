# Comprehensive Credit Risk Modelling

Statistical credit risk modeling, probability of default prediction

### Contributors
* <a href="https://github.com/sandesh-30"> Sandesh Gaikwad </a>
* <a href="https://github.com/Bhasin-IEOR">Shubham Bhasin</a>

## Preliminary Data Exploration & Processing
Used a dataset that was available on <a href= "https://www.kaggle.com/devanshi23/loan-data-2007-2014"> Kaggle </a> that relates to consumer loan. The raw data includes inormation on over 450,000 consumer loans issued between 2007 and 2014 with almost 75 features, including the current loan status and various attributes related to both borrowers and their payment behavior. 18 features with more than 80% of missing values. Given the high proportion of missing values, any technique to impute them will most likely result in inaccurate results
 
## Identify Target Variable
Based on the data exploration, the target variable appears to be loan_status. Based on domain knowledge, we will classify loans with the following loan_status values as being in default (or 0):
* Charged Off
* Default
* Late (31–120 days)
* Does not meet the credit policy. Status:Charged Off

All the other values will be classified as good (or 1)
 
 
## Data Split

Let us now split our data into the following sets: training (80%) and test (20%). We will perform Repeated Stratified k Fold testing on the training test to preliminary evaluate our model 
 
Splitting our data before any data cleaning or missing value imputation prevents any data leakage from the test set to the training set and results in more accurate model evaluation.
 
## Data Cleaning

Some necessary data cleaning tasks as follows:

Remove text from the emp_length column (e.g., years) and convert it to numeric

For all columns with dates: convert them to Python’s datetime format, create a new column as a difference between model development date and the respective date feature and then drop the original feature

## Feature Selection

Next up, we will perform feature selection to identify the most suitable features for our binary classification problem using the Chi-squared test for categorical features and ANOVA F-statistic for numerical features.

## One-Hot Encoding and Update Test Dataset

Next, we will create dummy variables of the four final categorical variables and update the test dataset through all the functions applied so far to the training dataset.
Note a couple of points regarding the way we create dummy variables:
We will use a particular naming convention for all variables: original variable name, colon, category name
Generally speaking, in order to avoid multicollinearity, one of the dummy variables is dropped through the drop_first parameter of pd.get_dummies. However, we will not do so at this stage as we require all the dummy variables to calculate the Weight of Evidence (WoE) and Information Values (IV) of our categories — more on this later. We will drop one dummy variable for each category later on

## WoE Binning and Feature Engineering

Creating new categorical features for all numerical and categorical variables based on WoE is one of the most critical steps before developing a credit risk model
WoE is a measure of the predictive power of an independent variable in relation to the target variable. It measures the extent a specific feature can differentiate between target classes, in our case: good and bad customers

A positive WoE means that the proportion of good customers is more than that of bad customers and vice versa for a negative WoE value.
 
## Information Value (IV)

IV is calculated as follows:

IV is only useful as a feature selection and importance technique when using a binary logistic regression model.
define a custom ‘transformer’ class using sci-kit learn’s BaseEstimator and TransformerMixin classes. Like other sci-kit learn’s ML models, this class can be fit on a dataset to transform it as per our requirements. Another significant advantage of this class is that it can be used as part of a sci-kit learn’s Pipeline to evaluate our training data using Repeated Stratified k-Fold Cross-Validation. Using a Pipeline in this structured way will allow us to perform cross-validation without any potential data leakage between the training and test folds
 
## Model Training

Finally, we come to the stage where some actual machine learning is involved. We will fit a logistic regression model on our training set and evaluate it using RepeatedStratifiedKFold. Note that we have defined the class_weight parameter of the LogisticRegression class to be balanced. This will force the logistic regression model to learn the model coefficients using cost-sensitive learning, i.e., penalize false negatives more than false positives during model training. Cost-sensitive learning is useful for imbalanced datasets, which is usually the case in credit scoring

## Prediction Time
It all comes down to this: Applied trained logistic regression model to predict the probability of default on the test set, which has not been used so far (other than for the generic data cleaning and feature selection tasks) 


## Main Highlights of this project

•	Applied logistic regression model to predict the Probability of Default (PD) in consumer loans to assess the risk involved

•	Evaluated weight of evidence (WOE) and Information value (IV) for feature engineering and selection

•	Performed feature selection using Chi-squared test for categorical features and ANOVA F-statistic for numerical features

* Reduced total number of features for model training by 45% using WoE ,IV ,F-statistic and P-value analysis
