from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd

cancer_data = load_breast_cancer()

model = LogisticRegression(solver='liblinear')

# We can see the available keys with the keys method.
# print(cancer_data.keys())

#  which gives a detailed description of the dataset.
# print(cancer_data['DESCR'])

# To see how many rows and cols it  has
# print(cancer_data['data'].shape)

# To get the featured name
# print(cancer_data['feature_names'])

# converting into pandas dataFrame
df = pd.DataFrame(cancer_data['data']  # Rows
                  , columns=cancer_data['feature_names'])

# prints the first five
# print(df.head())

# To know the target names ie. 'malignant' 'benign' 0,1
# print(cancer_data['target_names'])
# print(cancer_data['target'])

df['target'] = cancer_data['target']


# Take the feature name values
X = df[cancer_data["feature_names"]].values
y = df['target'].values

model.fit(X, y)

print(model.predict([X[0]]))
print(model.score(X, y))
