# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARITHA RAMESH
RegisterNumber:  212223100011

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")



*/
```

## Output:
![Screenshot 2025-03-07 183700](https://github.com/user-attachments/assets/7c7cd804-88a5-45ac-9a85-997b894a402c)


![Screenshot 2025-03-07 181328](https://github.com/user-attachments/assets/710d90fd-61e9-4f49-ab0e-97be36fdcf45)


![Screenshot 2025-03-07 183832](https://github.com/user-attachments/assets/87d929d9-6130-4027-8974-21661ed7435c)


![Screenshot 2025-03-07 183918](https://github.com/user-attachments/assets/49957d31-30df-48dc-ad32-d44125b56e2f)


![Screenshot 2025-03-07 183948](https://github.com/user-attachments/assets/3f962597-0ce8-4d0f-97d8-0d87062c869f)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
