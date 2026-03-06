import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier

X,y = make_circles(n_samples=200,noise=0.1)

model = MLPClassifier(hidden_layer_sizes=(3,3),learning_rate_init=0.03,max_iter=1000)
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm')
plt.title("Circular Data Classification")
plt.show()

print("Accuracy:",model.score(X,y))
