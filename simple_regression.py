import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn import datasets

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)


X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

input_size = n_features
output_size = 1
learning_rate = 0.01

model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


num_epochs = 1000

for epoch in range(num_epochs):
  y_pred = model(X)

  loss = criterion(y_pred, y)

  loss.backward()

  optimizer.step()

  optimizer.zero_grad()

  if (epoch+1) % 100 == 0:
    print(f"epoch {epoch+1}, loss {loss.item():.4f}")


predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X, predicted, "b")
plt.show()