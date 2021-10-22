import torch
from matplotlib import pyplot as plt
import numpy as np

arr = torch.load("res.pt")
arrd = arr[0]

data = arrd[-1]
data = [[int(data[0]), int(data[1])],[int(data[2]), int(data[3])]]
print(data)
labels = ["expert1", "expert2"]
dats = ["MNIST", "FMNIST"]

X = np.arange(2)
ind = np.arange(4)
fig, ax = plt.subplots()
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.set_ylabel('Input count')
ax.set_xticks([X[0]+0.125, X[1]+0.125])
ax.set_xticklabels(labels)
ax.legend(labels=dats)
ax.set_xlabel("All Experts")
ax.set_title("Gated input for each expert with both datasets")
plt.show()
