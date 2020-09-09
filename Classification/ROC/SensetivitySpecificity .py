from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
import pandas as pd

import plotlib as plt

sensitivity_score = recall_score


def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    # r[0] gives specificity and  r[1] gives snsitivity
    return r[0]


df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

model = LogisticRegression()
model.fit(X_train, y_train)

# Adjusting the threshhold
# y_pred = model.predict_proba(X_test)[:, 1] > 0.75
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()

# print("sensitivity:", sensitivity_score(y_test, y_pred))
# print("specificity:", specificity_score(y_test, y_pred))
