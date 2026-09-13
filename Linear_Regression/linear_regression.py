
#IMPORT THE LIBRARYS
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error,r2_score  
from sklearn.model_selection import train_test_split

 #2 CREATE THE STUDENT DATASET
data = {

 "Hours_Studied": [ 
  1.0, 1.5, 2.0, 2.5, 3.0, 
  3.5, 4.0, 4.5, 5.0, 5.5, 
  6.0, 6.5, 7.0, 7.5, 8.0, 
  8.5, 9.0, 9.5, 10.0, 10.5], 

  "Exam_Score": [
   52, 54, 57, 59, 62, 
   64, 66, 69, 71, 73, 
   75, 78, 80, 82, 84, 
   86, 88, 90, 92, 94]} 

df = pd.DataFrame(data)

#4 SEPARATE THE DATASET TO FEATURES AND LABELS X/Y 
X = df[["Hours_Studied"]] 
Y = df["Exam_Score"] 

#5 SPLIT THE DATASET TO TRAINING AND TESTING DATA 
X_train, X_test, Y_train, Y_test = train_test_split( 
X,   
Y,    
test_size = 0.20,   
random_state = 42 ) 

print(f"\n Training Sample:", len(X_train)) 
print(f"Testing Sample:", len(X_test))

 #6 IMPORT THE MODEL 
model = LinearRegression() 

 #7 TRAIN THE MODEL 
model.fit(X_train,Y_train) 

# VISUALIZE TRAINING DATA, TESTING DATA AND REGRESSION LINE
# Training data - BLUE
plt.scatter(
    X_train,
    Y_train,
    color="blue",
    label="Training Data"
)

# Testing data - YELLOW
plt.scatter(
    X_test,
    Y_test,
    color="yellow",
    edgecolor="black",
    label="Testing Data"
)

# Regression line - RED
plt.plot(
    X,
    model.predict(X),
    color="red",
    label="Regression Line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Training vs Testing Data")

plt.legend()
plt.show()

#EXTRACT THE COEFFICIENT AND THE INTERCEPT 
coefficient = model.coef_[0] 
intercept = model.intercept_ 

print(f"\n Coefficient:", coefficient) 
print(f"   Intercept:", intercept) 

print(f"\n Regression Equation:"     
      f"Exam_Score = {coefficient:.2f} * Hours_Studied + {intercept:.2f}") 

#MAKE PREDICTION 
prediction = model.predict(X_test) 
print(f"\n Prediction:", prediction) 

#EVALUATE THE MODEL
#  
mae = mean_absolute_error(Y_test, prediction) 
r2 = r2_score(Y_test, prediction) 

print(f"\n Model Evaluation:") 
print(f"-----------------------") 
print(f"Mean_Absolute_Error =", mae) 
print(f"R2_Score=", r2) 

print(f"\n............................")

#DEPLOY AND PREDICT NEW EXAM SCORE 
new_score = np.array([[7.5]])
new_predy = model.predict(new_score) 

print(f"\nNew_Exam_Score: {new_predy[0]:.2f}")


print(f"\n.............................")