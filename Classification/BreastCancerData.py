from sklearn.datasets import load_breast_cancer
import pandas as pd

cancer_data = load_breast_cancer()


# We can see the available keys with the keys method.
# print(cancer_data.keys())

#  which gives a detailed description of the dataset.
# print(cancer_data['DESCR'])

# To see how many rows and cols it  has
print(cancer_data['data'].shape)

# To get the featured name
print(cancer_data['feature_names'])

# converting into pandas dataFrame
df = pd.DataFrame(cancer_data['data']  # Rows
                  , columns=cancer_data['feature_names'])

# prints the first five
# print(df.head())

print(cancer_data['target'].shape)
