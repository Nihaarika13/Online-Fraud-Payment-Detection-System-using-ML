import pandas as pd
df=pd.read_csv('PS_20174392719_1491204439457_log.csv')
is_null = df.isnull().values.any()
if is_null:
    null_values = df.isnull().sum()
    print("Null values per column:")
    print(null_values)
    df.dropna(inplace = True)
    print("Removed all the null rows.")
    print("New dataset size: ", df.shape)
else:
   print("No null values found in the DataSet.")

is_duplicate=df.duplicated().any()
if is_duplicate:
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    df = df.drop_duplicates()
    print("Removed duplicate rows.")
    print("New dataset size: ", df.shape)
else:
  print("No duplicate rows found in the DataSet.")


# check for the types of payments 

distinct_types = df['type'].unique()
print("Types of payment: ", distinct_types)
types_dict = {v: i for i, v in enumerate(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])}
print("Types dictionary: ", types_dict)

df['type'] = df['type'].replace(types_dict)
print(df.head())

df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
print(df.head())
df=df.drop(['step','oldbalanceDest','newbalanceDest'],axis=1)
print(df.head())
df['isFraud'].value_counts()


# reduce the weightage of 'not fraud' data
from sklearn.utils import resample
majority = df[(df['isFraud']==0)] # not fraud data
minority = df[(df['isFraud']==1)] # fraud data

majority_downsampled = resample(majority, 
                                replace=False, 
                                n_samples=len(minority), 
                                random_state=71) #downsample the majority data

new_df = pd.concat([majority_downsampled, minority]) #concatenate the downsampled majority and minority
new_df = new_df.sample(frac=1).reset_index(drop=True) #shuffle the new dataframe

print("New dataframe size: ",new_df.shape)

print("\nNew Distribution")
print(new_df['isFraud'].value_counts())

new_df.head()

new_df.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned dataset saved.")