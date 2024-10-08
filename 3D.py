import os
from collections  import defaultdict
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def load_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    - file_path: The path to the pickle file.

    Returns:
    - loaded_data: The loaded data.
    """
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    # print(f'Data has been loaded from {file_path}')
    return loaded_data
# Generate example data
delta = np.linspace(0, 10, 1000)  # 1000 points for better resolution
gamma = np.linspace(0.01, 1, 1000)  # Avoid division by zero, start from 0.01
delta, gamma = np.meshgrid(delta, gamma)
eps = 0.01

# Correct z calculation with proper multiplication
z = (delta * gamma**5 * (1 - eps/gamma)) / (6 * (delta * gamma**2 + 1) * (1 + delta/gamma))

# Find the maximum value in z and its corresponding delta, gamma coordinates
max_index = np.unravel_index(np.argmax(z), z.shape)
max_delta = delta[max_index]
max_gamma = gamma[max_index]
max_z = z[max_index]

# Create the figure and a 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(gamma, delta, z, cmap='viridis')

# Label the point of the maximum ratio
ax.scatter(max_gamma, max_delta, max_z, color='hotpink', s=50)  # Red dot at the max point
ax.text(max_gamma-0.5, max_delta, max_z-0.001, f'({max_gamma:.2f}, {max_delta:.2f}, {max_z:.2f})', color='hotpink',fontsize=15)

# Add labels for axes
plt.xlabel('$\gamma$')
plt.ylabel('$\delta$')
ax.set_zlabel('Ratio,$\\alpha$')

# Save the plot as an image file
plt.savefig('3d_plot_max_label.pdf', dpi=500)
plt.savefig('3d_plot_max_label.png', dpi=500)

# Show the plot
plt.show()

# # Generate example data
# # Generate example data
# delta = np.linspace(0, 10, 1000)  # Correct linspace usage, 100 points
# gamma = np.linspace(0.01, 1, 1000)  # Avoid division by zero, start from 0.01
# delta, gamma = np.meshgrid(delta, gamma)
# eps = 0.1

# # Correct z calculation with proper multiplication
# z = (delta * gamma**5 * (1 - eps/gamma)) / (6 * (delta * gamma**2 + 1) * (1 + delta/gamma))

# # Create the figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface

# ax.plot_surface(gamma, delta, z, cmap='viridis')
# # ax.view_init(elev=0, azim=45)
# # ax.view_init()

# # Add labels
# # ax.set_xlabel('Gamma')
# # ax.set_ylabel('Delta')
# # ax.set_zlabel('Ratio')
# plt.xlabel('$\gamma$')
# plt.ylabel('$\delta$')
# ax.set_zlabel('Ratio')


# # Save the plot as an image file
# plt.savefig('3d_plot.pdf',dpi=500)
# plt.savefig('3d_plot.png',dpi=500)

# # Show the plot
# plt.show()