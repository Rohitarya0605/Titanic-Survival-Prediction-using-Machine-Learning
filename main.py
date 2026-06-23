import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("dataset/train.csv")

print("First 5 Rows:")
print(df.head())

# =========================
# STEP 2: Dataset Info
# =========================
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

# =========================
# STEP 3: Check Missing Values
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# STEP 4: Clean Missing Values
# =========================

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column
df = df.drop("Cabin", axis=1)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# =========================
# STEP 5: Encoding
# =========================

# Convert male/female to 0/1
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Convert Embarked to numeric columns
df = pd.get_dummies(df, columns=["Embarked"])

print("\nColumns After Encoding:")
print(df.columns)

print("\nFinal Dataset Preview:")
print(df.head())

# =========================
# STEP 6: Feature Selection
# =========================
df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)

print("\nColumns After Feature Selection:")
print(df.columns)

print("\nFinal Shape:")
print(df.shape)

# =========================
# STEP 7: Split Features & Target
# =========================
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# =========================
# STEP 8: Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# =========================
# STEP 9: Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

from sklearn.metrics import accuracy_score

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nPredictions:")
print(y_pred[:10])

print("\nAccuracy:")
print(accuracy)