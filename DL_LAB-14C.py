import numpy as np
import matplotlib.pyplot as plt

# Data (slightly changed values so output is different)
X = np.array([33, 52, 60, 48, 58, 54, 51, 40, 49, 53, 46, 55, 45, 57, 56, 49, 43, 61, 46, 39])
Y = np.array([32, 69, 63, 72, 86, 77, 80, 60, 74, 70, 56, 81, 63, 74, 82, 61, 83, 96, 50, 55])

w = 0
b = 0
lr = 0.0001
iterations = 500

costs = []

for i in range(iterations):
    y_pred = w * X + b
    cost = np.mean((Y - y_pred)**2)
    costs.append(cost)

    dw = (-2/len(X)) * np.sum(X*(Y-y_pred))
    db = (-2/len(X)) * np.sum(Y-y_pred)

    w = w - lr*dw
    b = b - lr*db

print("Estimated Weight:", w)
print("Estimated Bias:", b)

# Graph 1: Cost vs Iterations
plt.figure(figsize=(8,5))
plt.plot(costs,'r.')
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# Graph 2: Regression line
Y_pred = w*X + b

plt.figure(figsize=(8,5))
plt.scatter(X,Y,color='black',marker='*',label='Data Points')
plt.plot(X,Y_pred,'r--',label='Fitted Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
