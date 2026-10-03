import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Iris dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

print("Iris dataset loaded successfully!")
print(df.head())


# Dataset information
print("\nDataset shape:")
print(df.shape)

print("\nSpecies count:")
print(df["species"].value_counts())

print("\nDataset information:")
print(df.info())


# Features and target
X = df[iris.feature_names]
y = df["species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Create and train Decision Tree model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully!")


# Make predictions
y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)

print("\nActual values:")
print(y_test.values)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Model Accuracy: {:.2f}%".format(accuracy * 100))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# Plot Confusion Matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Iris Flower Classification - Confusion Matrix")
plt.show()


# Predict a new Iris flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=iris.feature_names
)

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted species:", iris.target_names[prediction[0]])