import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.arange(-10, 10, 0.2)   # changed input values

plt.plot(x, sigmoid(x), color='pink')
plt.title('Visualization of the Sigmoid Function')
plt.xlabel('Input Values')
plt.ylabel('Sigmoid Output')
plt.show()
