

# Topic - A model to predict customer churn for a subscription- based service or business.

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

churn_modelling_dataset = pd.read_csv('Churn_Modelling.csv')

# Counting the number of 1s in target value
churn_model_train_ones_count = churn_modelling_dataset['Exited'].value_counts().get(1,0)

# Randomly selecting the dataset whose value=0 equal to number of 1s
churn_model_zero_values = churn_modelling_dataset[churn_modelling_dataset['Exited'] == 0]
churn_random_zero_values = churn_model_zero_values.sample(n=churn_model_train_ones_count, random_state=42)

# Selecting the dataset whose value=1
churn_random_ones_values = churn_modelling_dataset[churn_modelling_dataset['Exited'] == 1]

# Concatenating the values of 0 and 1
balanced_churn_dataset = pd.concat([churn_random_ones_values, churn_random_zero_values])

# Shuffling the dataset with values 0 and 1
balanced_churn_dataset= balanced_churn_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# Extracting the features which is needed to train the model
feature_names_drop = ['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'Exited']
X = balanced_churn_dataset.drop(columns=feature_names_drop, axis=1)
Y = balanced_churn_dataset['Exited']

# Checking the null values
# null_val = X['Balance'].isnull()
# print(null_val)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}
for model_name, model in models.items():
    clf = Pipeline(steps=[('classifier', model)])
    clf.fit(X_train, Y_train)

    X_train_prediction = clf.predict(X_train)
    accuracy_of_train = accuracy_score(X_train_prediction, Y_train)

    X_test_prediction = clf.predict(X_test)
    accuracy_of_test = accuracy_score(X_test_prediction, Y_test)

    print(model_name)
    print(accuracy_of_train*100)
    print(accuracy_of_test*100)

    # The accuracy for the X_train and Y_train dataset for differnt models output:
    # Logistic Regression
    # 60.386621663086835
    # 60.24539877300613
    # Random Forest
    # 100.0
    # 66.99386503067485
    # Gradient Boosting
    # 74.65480208652961
    # 69.93865030674846

best_model = models['Gradient Boosting']  # Choose the best performing model
clf = Pipeline(steps=[('classifier', best_model)])
clf.fit(X_train, Y_train)
joblib.dump(clf, "best_churn_model.pkl")

print("Best model saved successfully")


input_data = (535,9,0,1,1,0,149892.8) # One sample input dataset whose target value should be 1

input_array = np.asarray(input_data)

input_array_reshaped = input_array.reshape(1,-1)


model = joblib.load('best_churn_model.pkl')
predictions = model.predict(input_array_reshaped)
if predictions[0] == 1:
    print("The customer will leave the bank")
else:
    print("The customer will not leave the bank")
