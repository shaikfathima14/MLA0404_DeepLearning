import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X,y = make_classification(n_samples=100,n_features=2,n_redundant=0,n_clusters_per_class=1)

model = LogisticRegression()
model.fit(X,y)

plt.scatter(X[:,0],X[:,1],c=y)

coef = model.coef_[0]
inter = model.intercept_

x_line = np.linspace(X[:,0].min(),X[:,0].max(),100)
y_line = -(coef[0]*x_line+inter)/coef[1]

plt.plot(x_line,y_line,'r--')
plt.title("Linear Separability")
plt.show()
