# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#CREATE THE CUSTOMER DATASET
data = { 

"Age": 
[22, 25, 28, 31, 35,
40, 45, 50, 55, 60,
23, 27, 30, 33, 37,
42, 47, 52, 58, 63,
21, 24, 29, 34, 39,
44, 49, 54, 59, 32 ],


"Income":
[25000, 32000, 38000, 45000, 52000,
60000, 70000, 80000, 90000, 95000,
28000, 35000, 42000, 48000, 55000,
65000, 72000, 85000, 88000, 100000,
22000, 30000, 40000, 50000, 58000,
68000, 76000, 82000, 92000, 46000 ],


"Site_Visits":
[1, 2, 3, 4, 5,
6, 5, 7, 8, 6,
4, 5, 6, 2, 3,
2, 3, 4, 2, 3,
6, 7, 1, 7, 8,
7, 8, 1, 5, 1 ],


"Bought":
[0, 0, 0, 1, 1,
1, 1, 1, 1, 1,
0, 0, 1, 0, 1,
1, 1, 1, 1, 1,
0, 0, 0, 1, 1,
1, 1, 0, 1, 0 ]
}


# CREATE DATAFRAME
df = pd.DataFrame(data)


# INSPECT THE DATA
print(f"\n First Five Rows:")
print(df.head())

print(f"\n Dataset Information:")
print(df.info())

print(f"\n Missing Values:")
print(df.isnull().sum())

print(f"\n Dataset Summary:")
print(df.describe())


# SEPARATE FEATURES AND TARGET 
X = df[[ "Age", "Income", "Site_Visits"]]
Y = df["Bought"]

# SPLIT DATA INTO TRAINING AND TESTING
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.20,
    random_state = 42
)

print(f"\n Training Sample:", len(X_train))
print(f" Testing Sample:", len(X_test))


# CREATE THE CLASSIFICATION MODEL
model = DecisionTreeClassifier()

# TRAIN THE MODEL
model.fit(X_train, Y_train)

# MAKE PREDICTIONS
predictions = model.predict(X_test)

print(f"\n Actual Value:")
print(Y_test.values)

print(f"\n Predictated Value:")
print(predictions)


# EVALUATE THE MODEL
accuracy = accuracy_score(Y_test, predictions)

print(f"\n Model Evaluation:")
print("........................")
print("Accuracy:", accuracy)


# CONFUSION MATRIX
cm = confusion_matrix(Y_test, predictions)

print(f"\n Confusion Matrix:")
print(".....................")
print(cm)



# VISUALIZE THE DATA
no_purchase = df[df["Bought"] == 0]
purchase = df[df["Bought"] == 1]


plt.scatter(
    no_purchase["Income"],
    no_purchase["Site_Visits"],
    color = "red",
    label = "No_Purchase"
)


plt.scatter(
    purchase["Income"],
    purchase["Site_Visits"],
    color = "Green",
    label = "purchase"
)


plt.xlabel("Income")
plt.ylabel("Site_Visits")
plt.title("Customer Purchase Classification")
plt.legend()
plt.show()


# PREDICT A NEW CUSTOMER
new_customer = pd.DataFrame({
    "Age":[35],
    "Income":[55000],
    "Site_Visits":[5]
})

new_predy = model.predict(new_customer)

print(f"\n.................................")
if new_predy[0] == 1:
    print("\n New Customer Prediction : Purchase")
else:
    print(f"\nNew Customer Prediction : No Purchase")
