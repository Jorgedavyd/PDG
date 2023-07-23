import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

#point creation

def get_map(full_dataframe, model):
    full_dataframe['Probability'] = full_dataframe[:19]





# Generate random data for demonstration
num_points = 100
paraguay_lats = np.random.uniform(low=-27, high=-19, size=num_points)
paraguay_lons = np.random.uniform(low=-63, high=-54, size=num_points)
values = np.random.uniform(low=0, high=1, size=num_points)

# Create a Basemap instance
m = Basemap(projection='cyl', resolution='l',
            llcrnrlat=-27, urcrnrlat=-19,
            llcrnrlon=-63, urcrnrlon=-54)

# Create a figure and axes
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Draw coastlines and borders
m.drawcoastlines()
m.drawcountries()

# Convert latitudes and longitudes to map coordinates
x, y = m(paraguay_lons, paraguay_lats)

# Plot the heat map using scatter plot
sc = m.scatter(x, y, c=values, cmap='hot', edgecolors='k', linewidth=0.5)

# Add a colorbar
plt.colorbar(sc, label="Heat")

# Set plot title
plt.title("Paraguay Heat Map")

plt.show()

