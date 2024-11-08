import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('hearing_loss_full_data2set.csv')

print("Initial Data Sample:")
print(df.head())

print("\nChecking for missing values:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


print("\nData after encoding and scaling:")
print(df.head())

