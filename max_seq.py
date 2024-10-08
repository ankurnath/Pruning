import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Sample data
x = np.arange(10)
y = np.exp(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# Set the y-axis tick locations
ax.yaxis.set_major_locator(ticker.FixedLocator([10, 100, 1000, 10000]))

plt.show()