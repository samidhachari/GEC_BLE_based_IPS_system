import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'Zone123.csv')

# Convert device serial number to numerical values
le = LabelEncoder()
data['deviceSerialNumber'] = le.fit_transform(data['deviceSerialNumber'])

# Split the data into features and target
X = data[['deviceSerialNumber', 'beaconId', 'rssiCh37', 'rssiCh38', 'rssiCh39', 'seqNo']].astype(float)
y = data['Block']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=72)

# Build the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
clf.fit(X_train, y_train)

# Predict the test set results
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
