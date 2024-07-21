#importing pandas
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


X = df.drop(['isFraud'], axis=1)
Y = df['isFraud']

print(X.shape, Y.shape)

#import train-test split function
from sklearn.model_selection import train_test_split

#first split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.05, random_state=71
)

#further split the training data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=71)

#show the data sizes
print(f"X Train: {X_train.shape}, Y Train: {Y_train.shape}")
print(f"X Validation: {X_val.shape}, Y Validation: {Y_val.shape}")
print(f"X Test: {X_test.shape}, Y Test: {Y_test.shape}")


#import few well known classification models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#import evalutation metrics 
from sklearn.metrics import accuracy_score,classification_report
#store the model objects in a list
models = [LogisticRegression(),
          SVC(kernel='rbf', probability=True),
          RandomForestClassifier()]
#train each model on the training data and check their accuracy on training and validation sets
for i in range(len(models)):
    models[i].fit(X_train, Y_train)
    
    train_accuracy = accuracy_score(Y_train, models[i].predict(X_train))
    validation_accuracy = accuracy_score(Y_val, models[i].predict(X_val))

    print(f"\n{models[i]}")
    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", validation_accuracy)

    #choose the final model
rf = models[2]

#generate classification report on validation data
train_accuracy = accuracy_score(Y_train, rf.predict(X_train))
validation_accuracy = accuracy_score(Y_val, rf.predict(X_val))

Y_pred = rf.predict(X_val)

print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", validation_accuracy)
print("\nClassification report for model:")
print(classification_report(Y_pred, Y_val))

#importing os and pickle
import os
import pickle

# Specify the folder path and report filename
folder_path = "bin"
report_filename = "classification_report.pkl"

#create file path
file_path = os.path.join(folder_path, report_filename)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

report = classification_report(Y_pred, Y_val, output_dict=True)

#save the classification report
with open(file_path, 'wb') as f:
  pickle.dump(report, f)

print(f"Classification report saved at {file_path}")

#import joblib
import joblib

# Specify the folder path and model filename
folder_path = "bin"
model_filename = "fraud_detection_model.joblib"

#create file path
file_path = os.path.join(folder_path, model_filename)

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Save the model to the specified file using joblib.dump
joblib.dump(rf, file_path)

print(f"Model saved at {file_path}")

rf_model = joblib.load('bin/fraud_detection_model.joblib')
#perform prediction on test data
Y_test_pred = rf_model.predict(X_test)
#plot confusion metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np 

cm = confusion_matrix(Y_test_pred, Y_test)

# Normalize the confusion matrix (optional)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, cmap=plt.cm.Blues)  # Choose your preferred colormap

# Add labels and title
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')

# Add text labels for each cell (optional)

labels = [0,1]

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

plt.xticks(np.arange(len(labels)), labels, rotation=45)  # Replace with your target labels
plt.yticks(np.arange(len(labels)), [0,1])
plt.tight_layout()
plt.show()