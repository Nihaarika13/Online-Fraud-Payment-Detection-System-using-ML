import pandas as pd
df=pd.read_csv("Retrieved_File.csv")

#import Standard scaler
from sklearn.preprocessing import StandardScaler
#list the numerical columns to standardise
columns_to_normalize = ['type', 'amount','oldbalanceOrg','newbalanceOrig']
#Initialise the Standard scaler object
scaler = StandardScaler()
#Normalise the data and show the result
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
print(df.head())


#import the os and pickle
import os
import pickle

#create the folder name and scaler file name
folder_path = "bin"
scaler_filename = "scaler.pkl"

#create the file path
file_path = os.path.join(folder_path, scaler_filename)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Save the scaler object to the file
with open(file_path, 'wb') as f:
  pickle.dump(scaler, f)

print(f"Standard scaler saved to {file_path}")