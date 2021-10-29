import matplotlib.pyplot as plt
import numpy as np


window = 20

with open("models/camel_regenerating_e5000_h300_loss.txt") as f:
# with open("models/camel_regenerating_e5000_h300_loss.txt") as f:
    data = f.read().split("\n")


losses = np.array([ float(x) for x in data ])
n = losses.shape[0]

avg_losses = np.zeros_like(losses)

for i in range(0, n):
    avg_losses[i] = losses[i-min(window, i):i+window].sum() / window

plt.plot(np.arange(avg_losses.shape[0]), avg_losses)
plt.show()
print(avg_losses[-1], losses.sum() / n)

with open("models/camel_conv_regenerating_e5000_h300_loss.txt") as f:
# with open("models/camel_regenerating_e5000_h300_loss.txt") as f:
    data = f.read().split("\n")


losses = np.array([ float(x) for x in data ])
n = losses.shape[0]

avg_losses = np.zeros_like(losses)

for i in range(0, n):
    avg_losses[i] = losses[i-min(window, i):i+window].sum() / window


plt.plot(np.arange(avg_losses.shape[0]), avg_losses)
plt.show()
print(avg_losses[-1], losses.sum() / n)



