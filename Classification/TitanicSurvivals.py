import pandas as pd
import numpy as np

# Read the file
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

# Set new col with boolean values True if male False if female
df['male'] = df['sex'] == 'male'

# Features
X = df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].values

# The target
Y = df[['Survived']].values

print(x)
print(y)
