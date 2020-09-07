import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Read the file
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

# Declaring the Logistic Regression class
model = LogisticRegression()

# Set new col with boolean values True if male False if female
df['male'] = df['Sex'] == 'male'

# Features
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].values

# The target
y = df['Survived'].values

model.fit(X, y)
model.predict(X)
# model.coef_ are an array with two values the first is the coeficcient of "X" and the second one is the coeficient of "Y"
# The more of X you have the more numbers you get
# model.intercept_ is the constant
# print(model.coef_, model.intercept_)

# Predicting an existing model
# print(model.predict([[3, True, 22.0, 1, 0, 7.25]]))
# print(model.predict(X[:5]))
# print(y[:5])
# Array that has the predicted X values
y_pred = model.predict(X)

# To check how many ar true
# print((y == y_pred).sum())

# To get the precentage y.shape[0] is the full number
# print((y == y_pred).sum() / y.shape[0])

# Printin the score
print(model.score(X, y))
