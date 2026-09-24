import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

data = {
    'Age': [25, np.nan, 28, 35, np.nan, 40],
    'Salary': [50000, 54000, np.nan, 58000, 60000, np.nan],
    'Department': ['HR', 'HR', 'IT', np.nan, 'Finance', 'Finance']
}

df = pd.DataFrame(data)

print("Original Data:\n", df)

print("\nMissing Values (Before Imputation):")
print(df.isnull().sum())

mean_imputer = SimpleImputer(strategy='mean')

df['Age_mean'] = mean_imputer.fit_transform(df[['Age']]).ravel()
df['Salary_mean'] = mean_imputer.fit_transform(df[['Salary']]).ravel()

median_imputer = SimpleImputer(strategy='median')

df['Age_median'] = median_imputer.fit_transform(df[['Age']]).ravel()
df['Salary_median'] = median_imputer.fit_transform(df[['Salary']]).ravel()

freq_imputer = SimpleImputer(strategy='most_frequent')

df['Department_mode'] = freq_imputer.fit_transform(df[['Department']]).ravel()

print("\nMissing Values (After Imputation on New Columns):")
print(df[['Age_mean', 'Salary_mean', 'Age_median', 'Salary_median', 'Department_mode']].isnull().sum())

df.to_csv("imputed_data.csv", index=False)

print("\nFinal Data exported to 'imputed_data.csv'")

print("\nFinal DataFrame:\n", df)