import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier

X,y = make_moons(n_samples=200,noise=0.2)

model = MLPClassifier(hidden_layer_sizes=(3,3),learning_rate_init=0.03,max_iter=1000)
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm')
plt.title("Two Class Neural Network")
plt.show()

print("Accuracy:",model.score(X,y))
