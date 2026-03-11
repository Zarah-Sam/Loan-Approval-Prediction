import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("loan_data.csv")

# Fill missing values
data = data.ffill()

# Encode categorical columns
label = LabelEncoder()

categorical = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']

for col in categorical:
    data[col] = label.fit_transform(data[col])

# Features and target
X = data.drop(['Loan_ID','Loan_Status'], axis=1)
y = data['Loan_Status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train,y_train)

# Save model
joblib.dump(model,"loan_model.pkl")
joblib.dump(scaler,"scaler.pkl")

print("Model trained successfully")