
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv("salary.csv")

print("Dataset Shape:", dataset.shape)
print("\nFirst 5 Records:\n", dataset.head())

dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1})
print("\nUpdated Dataset:\n", dataset.head())


X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

error = []

for i in range(1, 40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 5))
plt.plot(range(1, 40), error, marker='o')
plt.title("Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Mean Error")
plt.show()

model = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
model.fit(X_train, y_train)

age = int(input("Enter Employee Age: "))
edu = int(input("Enter Education Level (numeric): "))
cg = int(input("Enter Capital Gain: "))
wh = int(input("Enter Hours per Week: "))

new_employee = [[age, edu, cg, wh]]
prediction = model.predict(sc.transform(new_employee))

if prediction[0] == 1:
    print("\n✅ Employee may earn ABOVE 50K")
else:
    print("\n❌ Employee may earn BELOW 50K")

y_pred = model.predict(X_test)

print("\nPredicted vs Actual:")
print(np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1))

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\nConfusion Matrix:")
print(cm)

print("\nModel Accuracy: {:.2f}%".format(accuracy))
