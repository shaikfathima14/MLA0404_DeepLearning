import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

X,y = make_blobs(n_samples=300,centers=3,random_state=1)

model = MLPClassifier(hidden_layer_sizes=(2,2),learning_rate_init=0.01,max_iter=1000)
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y,cmap='viridis')
plt.title("Multi Class Neural Network")
plt.show()

print("Accuracy:",model.score(X,y))
