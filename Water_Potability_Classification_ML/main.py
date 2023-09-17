import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acs
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from warnings import filterwarnings
filterwarnings('ignore')

from zipfile import ZipFile


# Data Extraction
with ZipFile('water.zip') as water_data:
    water_data.extractall()


# Data Preprocessing & Feature Analysis
df = pd.read_csv('water_potability.csv')

print(df.info(),
      df.isnull().sum(),
      df.dtypes
      )

df = df.dropna()

print(df.info(),
      df.isnull().sum()
      )

heatmap = sns.heatmap(df.corr(), annot= True)
plt.savefig('heatmap.png')
plt.show()


# Train Test Split
X = df.drop('Potability', axis= 1)
Y = df['Potability']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    shuffle= True,
                                                    random_state= 24,
                                                    test_size= 0.25
                                                    )


# Model Training
m = XGBClassifier()

m.fit(X_train, Y_train)

pred_train = m.predict(X_train)
print(f'Train Accuracy is : {acs(Y_train, pred_train)}')

pred_test = m.predict(X_test)
print(f'Test Accuracy is : {acs(Y_test, pred_test)}')