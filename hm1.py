import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train-50(1000)-100.csv')

# Split to X and Y
matrix = np.array(data,'float')
x = matrix[:,:100]
y = matrix[:,100]

# plot
plt.plot(x,y,'bo')
plt.show()

# Training and Test
num = int(len(data)*0.8)

x_train = x[:num]
y_train = y[:num]
x_test = x[num:]
y_test = y[num:]

# (for testing the W) Calculating the W with L2, for only Lambda = 10:
lam = 10
W10 = np.linalg.inv(x_train.transpose().dot(x_train)+lam).dot(x_train.transpose()).dot(y_train)


# Calculating the W with L2, and Lambda loop from 1 to 150:
all_W = []
for lam in range(1,151):
    W = np.linalg.inv(x_train.transpose().dot(x_train)+lam).dot(x_train.transpose()).dot(y_train)
    all_W.append(W)


# Calculating MSE:
Ew = 1
