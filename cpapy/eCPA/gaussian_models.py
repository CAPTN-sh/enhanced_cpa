"""

Use Gaussian models to compute an enhanced version of the Closest Point of Approach (eCPA).

Author: Tom Beyer

"""

import sys
sys.path.append("../../Code")

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from eCPA_for_gaussian_models import compute_ecpa
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# # To avoid a Mac specific issue in Terminal printouts
# f = open("/dev/null", "w")
# os.dup2(f.fileno(), 2)
# f.close()


"""
Parameters and options
"""

# Method of computing the Gaussian model (Gaussian Kernel Density Estimation or Gaussian Mixture Model)
method = 'gmm'  # 'kde' or 'gmm'
# Plotting yes/no?
plotting = True
# Gaussian Mixture Model number of components (if GMM is chosen)
n_components = 1

# Define the area of interest
min_x, max_x = 0, 300
min_y, max_y = 0, 300
num_points = 3000

# Dynamic courses (otherwise everything is linear)?
dynamic = True
# Wind data included? (only possible in linear/non-dynamic mode)
wind_enabled = False
# Current data included? (only possible in linear/non-dynamic mode)
current_enabled = False
# Dynamic prediction method (double_exponential, triple_exponential, polynomial)
prediction_method = 'double_exponential'
# Dynamic interpolation method (linear, quadratic, cubic, cubic_spline, cubic_hermite_spline, pchip, akima)
interpolation_method = 'akima'
# Dynamic course forecast steps
forecast_steps = 2
# How many past_timesteps should go into the dynamic prediction?
past_timesteps = 3
# Shall an approximation of the real DCPA (given the ships dimensions) be included in the result?
r_dcpa = True
# Shall the Obstacle Zone by Target (OZT) method be activated (only possible in linear mode!)?
ozt = False

# Define the security radius for the vessels (one point is approximately 14 meters)
security_radius = 7

# Data from the vessels involved, ordered from 
# (position [coordinates], COG [degrees], SOG [knots], heading [degrees], timestamp, wind influence, current influence, distances relative to AIS antenna)
ship1 = [{'x': 68, 'y': 65, 'COG': 10, 'SOG': 9.3, 'heading': 10, 'timestamp': 0.0, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1}, 
         {'x': 75, 'y': 100, 'COG': 20, 'SOG': 9.5, 'heading': 20, 'timestamp': 3.82, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1}, 
         {'x': 100, 'y': 150, 'COG': 35, 'SOG': 10, 'heading': 35, 'timestamp': 9.56, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1}]
ship2 = [{'x': 270, 'y': 220, 'COG': 325, 'SOG': 8.2, 'heading': 350, 'timestamp': 0.5, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1},
         {'x': 250, 'y': 250, 'COG': 280, 'SOG': 9.4, 'heading': 285, 'timestamp': 4.8, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1}, 
         {'x': 200, 'y': 250, 'COG': 235, 'SOG': 10, 'heading': 240, 'timestamp': 10.02, 'wind_factor': 0.3, 'current_factor': 0.1, 'bow': 3, 'stern': 2, 'portside': 1, 'starboard': 1}]

# ship3 = [{'x': 160, 'y': 232, 'COG': 209.74488129694222, 'SOG': 0.3, 'heading': 209.74488129694222, 'timestamp': 0.0, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 4.0, 'stern': 4.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 156, 'y': 225, 'COG': 209.74488129694222, 'SOG': 0.3, 'heading': 209.74488129694222, 'timestamp': 57.003214257332026, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 4.0, 'stern': 4.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 152, 'y': 218, 'COG': 209.74488129694222, 'SOG': 0.3, 'heading': 209.74488129694222, 'timestamp': 114.00641157081513, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 4.0, 'stern': 4.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 148, 'y': 211, 'COG': 209.74488129694222, 'SOG': 0.3, 'heading': 209.74488129694222, 'timestamp': 171.0095878808715, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 4.0, 'stern': 4.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 136, 'y': 206, 'COG': 247.38013505195957, 'SOG': 0.3, 'heading': 247.38013505195957, 'timestamp': 228.01273748935458, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 4.0, 'stern': 4.0, 'portside': 0.5, 'starboard': 0.5}]
# ship1 = [{'x': 188, 'y': 154, 'COG': 338.1985905136482, 'SOG': 0.45, 'heading': 338.1985905136482, 'timestamp': 0.0, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 2.0, 'stern': 2.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 182, 'y': 169, 'COG': 338.1985905136482, 'SOG': 0.45, 'heading': 338.1985905136482, 'timestamp': 57.003214257332026, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 2.0, 'stern': 2.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 176, 'y': 184, 'COG': 338.1985905136482, 'SOG': 0.45, 'heading': 338.1985905136482, 'timestamp': 114.00641157081513, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 2.0, 'stern': 2.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 170, 'y': 199, 'COG': 338.1985905136482, 'SOG': 0.45, 'heading': 338.1985905136482, 'timestamp': 171.0095878808715, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 2.0, 'stern': 2.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 170, 'y': 215, 'COG': 0.0, 'SOG': 0.45, 'heading': 0.0, 'timestamp': 228.01273748935458, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 2.0, 'stern': 2.0, 'portside': 0.5, 'starboard': 0.5}]
# ship4 = [{'x': 214, 'y': 258, 'COG': 204.44395478041653, 'SOG': 0.32, 'heading': 204.44395478041653, 'timestamp': 0.0, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 5.0, 'stern': 5.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 209, 'y': 247, 'COG': 204.44395478041653, 'SOG': 0.32, 'heading': 204.44395478041653, 'timestamp': 57.003214257332026, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 5.0, 'stern': 5.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 204, 'y': 236, 'COG': 204.44395478041653, 'SOG': 0.32, 'heading': 204.44395478041653, 'timestamp': 114.00641157081513, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 5.0, 'stern': 5.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 199, 'y': 225, 'COG': 204.44395478041653, 'SOG': 0.32, 'heading': 204.44395478041653, 'timestamp': 171.0095878808715, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 5.0, 'stern': 5.0, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 194, 'y': 214, 'COG': 204.44395478041653, 'SOG': 0.32, 'heading': 204.44395478041653, 'timestamp': 228.01273748935458, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 5.0, 'stern': 5.0, 'portside': 0.5, 'starboard': 0.5}]
# ship2 = [{'x': 96.0, 'y': 176.0, 'COG': 74.57783868126131, 'SOG': 0.26, 'heading': 74.57783868126131, 'timestamp': 0.0, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 0.5, 'stern': 0.5, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 113.03365051847561, 'y': 180.69893807406226, 'COG': 74.57783868126131, 'SOG': 0.3099981069905093, 'heading': 74.57783868126131, 'timestamp': 57.003214257332026, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 0.5, 'stern': 0.5, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 130.0672896829825, 'y': 185.39787301599517, 'COG': 74.57783868126131, 'SOG': 0.30999787802792794, 'heading': 74.57783868126131, 'timestamp': 114.00641157081513, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 0.5, 'stern': 0.5, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 147.1009145492973, 'y': 190.09680401359927, 'COG': 74.57783868126131, 'SOG': 0.3099975860579464, 'heading': 74.57783868126131, 'timestamp': 171.0095878808715, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 0.5, 'stern': 0.5, 'portside': 0.5, 'starboard': 0.5}, 
#          {'x': 164.13452085497718, 'y': 194.7957298910282, 'COG': 74.57783868126131, 'SOG': 0.3099972009230098, 'heading': 74.57783868126131, 'timestamp': 228.01273748935458, 'wind_factor': 0.0, 'current_factor': 0.0, 'bow': 0.5, 'stern': 0.5, 'portside': 0.5, 'starboard': 0.5}]

# Wind data (speed [m/s], direction [degrees]) [optional; only linear mode]
wind = [{'speed': 6.67, 'direction': 110}, 
        {'speed': 6.67, 'direction': 110}]

# Current data (speed [cm/s], direction [degrees]) [optional; only linear mode]
current = [{'speed': 1.3, 'direction': 200}, 
           {'speed': 1.3, 'direction': 200}]


"""
Functions
"""

# Calculates the Euclidean distance between two two-dimensional points.
def distance(point1, point2):

    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


"""
Compute the eCPA by Gaussian models
"""

# Compute the eCPA with different dynamic methods to collect data points for the Gaussian models
# Initialize the results
eCPA_results = []
time_to_cpa_results = []
cpa_distance_results = []
r_dcpa_results = []

# Do the computation
prediction_methods = ['double_exponential', 'triple_exponential', 'polynomial']
interpolation_methods = ['linear', 'quadratic', 'cubic', 'cubic_spline', 'cubic_hermite_spline', 'pchip', 'akima']
for prediction_method in prediction_methods:
    for interpolation_method in interpolation_methods:
        eCPA_temp, time_to_cpa_temp, cpa_distance_temp, r_dcpa_temp = compute_ecpa(
            dynamic=True,
            wind_enabled=False,
            current_enabled=False,
            prediction_method=prediction_method,
            interpolation_method=interpolation_method,
            forecast_steps=forecast_steps,
            past_timesteps=past_timesteps,
            r_dcpa=r_dcpa,
            ozt=False,
            ship1=ship1,
            ship2=ship2,
            wind=wind,
            current=current,
            security_radius=security_radius,
        )
        eCPA_results.append(eCPA_temp)
        time_to_cpa_results.append(time_to_cpa_temp)
        cpa_distance_results.append(cpa_distance_temp)
        r_dcpa_results.append(r_dcpa_temp)

# Compute the eCPA with the default/linear method (eventually including wind or current data if enabled)
eCPA_temp, time_to_cpa_temp, cpa_distance_temp, r_dcpa_temp = compute_ecpa(
    dynamic=False,
    wind_enabled=wind_enabled,
    current_enabled=current_enabled,
    prediction_method=prediction_method,
    interpolation_method=interpolation_method,
    forecast_steps=forecast_steps,
    past_timesteps=past_timesteps,
    r_dcpa=r_dcpa,
    ozt=False,
    ship1=ship1,
    ship2=ship2,
    wind=wind,
    current=current,
    security_radius=security_radius,
)
eCPA_results.append(eCPA_temp)
time_to_cpa_results.append(time_to_cpa_temp)
cpa_distance_results.append(cpa_distance_temp)
r_dcpa_results.append(r_dcpa_temp)

# Compute the Euclidean distance between the CPA by linear method and the eCPAs by 
# dynamic methods to determine the maximum distance
max_distance = max(distance(eCPA, eCPA_results[-1]) for eCPA in eCPA_results[:-1])

# Compute the eCPA with the default/linear method including the OZT method
ozt_line, time_to_cpa_temp, cpa_distance_temp, r_dcpa_temp = compute_ecpa(
    dynamic=False,
    wind_enabled=wind_enabled,
    current_enabled=current_enabled,
    prediction_method=prediction_method,
    interpolation_method=interpolation_method,
    forecast_steps=forecast_steps,
    past_timesteps=past_timesteps,
    r_dcpa=r_dcpa,
    ozt=True,
    ship1=ship1,
    ship2=ship2,
    wind=wind,
    current=current,
    security_radius=security_radius,
)

# Check if the OZT line makes geometric sense by computing the Euclidean distance between the CPA and the OZT line
# and comparing it to the maximum distance between the standard linear CPA and the eCPAs by dynamic methods
if ozt_line is not None:
    distance_to_ozt1 = distance(eCPA_results[-1], ozt_line[0])
    distance_to_ozt2 = distance(eCPA_results[-1], ozt_line[1])
    if distance_to_ozt1 > max_distance or distance_to_ozt2 > max_distance:
        ozt_line = None

# Check if it was geometrically possible to compute the OZT line
if ozt_line is not None:
    time_to_cpa_results.append(time_to_cpa_temp)
    cpa_distance_results.append(cpa_distance_temp)
    r_dcpa_results.append(r_dcpa_temp)

    # Sample points from the OZT line, roughly oriented on a Gaussian distribution
    # Calculate the midpoint of the ozt_line
    midpoint_x = (ozt_line[0][0] + ozt_line[1][0]) / 2
    midpoint_y = (ozt_line[0][1] + ozt_line[1][1]) / 2
    midpoint = (midpoint_x, midpoint_y)

    # Calculate the midpoints of every half of the ozt_line
    midpoint1_x = (ozt_line[0][0] + midpoint[0]) / 2
    midpoint1_y = (ozt_line[0][1] + midpoint[1]) / 2
    midpoint1 = (midpoint1_x, midpoint1_y)
    midpoint2_x = (ozt_line[1][0] + midpoint[0]) / 2
    midpoint2_y = (ozt_line[1][1] + midpoint[1]) / 2
    midpoint2 = (midpoint2_x, midpoint2_y)

    # "Sample" points from the OZT line (2/3 of the points are the midpoint, 1/3 are the midpoints of the halves)
    for _ in range(4):
        eCPA_results.append(midpoint)
    eCPA_results.append(midpoint1)
    eCPA_results.append(midpoint2)

# Format the eCPA points
eCPA_points = np.array(eCPA_results)
print(f'eCPA_points: {eCPA_points}')
# Fit the model to the data
if method == 'kde':
    # Fit the KDE to the data
    kde = gaussian_kde(eCPA_points.T)
elif method == 'gmm':
    # Fit the GMM to the data
    gmm = GaussianMixture(n_components=n_components)  # for example, we assume 3 clusters
    gmm.fit(eCPA_points)

# Create a grid of points
x = np.linspace(min_x, max_x, num_points)
y = np.linspace(min_y, max_y, num_points)
X, Y = np.meshgrid(x, y)
grid_points = np.vstack([X.ravel(), Y.ravel()])

# Evaluate the Gaussian model at each point on the grid
if method == 'kde':
    # Evaluate the KDE at each point on the grid
    Z = kde(grid_points).reshape(X.shape)
elif method == 'gmm':               
    # Evaluate the GMM at each point on the grid
    Z = np.exp(gmm.score_samples(grid_points.T)).reshape(X.shape)

# Find the index of the maximum value in the flattened array
max_index_flat = np.argmax(Z)

# Convert the index back to 2D coordinates
max_index_2d = np.unravel_index(max_index_flat, Z.shape)

# Get the x and y coordinates of the maximum point
max_x = X[max_index_2d]
max_y = Y[max_index_2d]

# Compute additional eCPA values (TCPA, DCPA, RCPA) using the chosen model
# Process time_to_cpa_results
if method == 'kde':
    kde_time_to_cpa = gaussian_kde(time_to_cpa_results)
    max_time_to_cpa = np.max(time_to_cpa_results)
elif method == 'gmm':
    gmm_time_to_cpa = GaussianMixture(n_components=n_components)
    gmm_time_to_cpa.fit(np.array(time_to_cpa_results).reshape(-1, 1))
    max_time_to_cpa = gmm_time_to_cpa.means_[np.argmax(gmm_time_to_cpa.weights_)][0]

# Process cpa_distance_results
if method == 'kde':
    kde_cpa_distance = gaussian_kde(cpa_distance_results)
    max_cpa_distance = np.max(cpa_distance_results)
elif method == 'gmm':
    gmm_cpa_distance = GaussianMixture(n_components=n_components)
    gmm_cpa_distance.fit(np.array(cpa_distance_results).reshape(-1, 1))
    max_cpa_distance = gmm_cpa_distance.means_[np.argmax(gmm_cpa_distance.weights_)][0]

# Process r_cpa_results
if r_dcpa:
    if method == 'kde':
        kde_r_dcpa = gaussian_kde(r_dcpa_results)
        max_r_dcpa = np.max(r_dcpa_results)
    elif method == 'gmm':
        gmm_r_dcpa = GaussianMixture(n_components=n_components)
        gmm_r_dcpa.fit(np.array(r_dcpa_results).reshape(-1, 1))
        max_r_dcpa = gmm_r_dcpa.means_[np.argmax(gmm_r_dcpa.weights_)][0]

# Plot the result
if plotting:
    # Print the maximum point
    if method == 'kde':
        print(f"The maximum point in the Gaussian KDE is at ({max_x}, {max_y})")
    elif method == 'gmm':
        print(f"The maximum point in the GMM is at ({max_x}, {max_y})")

    # Plot the result
    plt.figure(figsize=(10, 5))  # create a new figure
    for i in range(1, 3):  # loop to create two subplots
        plt.subplot(1, 2, i)  # create a new subplot
        # Set the limits of the x and y axes
        if i == 1:
            plt.xlim(0, 300)
            plt.ylim(0, 300)
        else:  # zoom in around the maximum point in the second plot
            plt.xlim(max_x - 20, max_x + 20)
            plt.ylim(max_y - 20, max_y + 20)
        # img = plt.imread("foerde_cutout.png")
        # plt.imshow(img, extent=[0, 300, 0, 300])

        # Plot the ships
        plt.plot(ship1[-1]['x'], ship1[-1]['y'], "ob", zorder=2)
        plt.plot(ship2[-1]['x'], ship2[-1]['y'], "og", zorder=2)

        # Plot the ships speed and heading as arrow
        plt.arrow(ship1[-1]['x'], ship1[-1]['y'], ship1[-1]['SOG'] * math.sin(math.radians(ship1[-1]['heading'])),
                    ship1[-1]['SOG'] * math.cos(math.radians(ship1[-1]['heading'])),
                    head_width=5, head_length=5, color='b', zorder=2)
        plt.arrow(ship2[-1]['x'], ship2[-1]['y'], ship2[-1]['SOG'] * math.sin(math.radians(ship2[-1]['heading'])),
                    ship2[-1]['SOG'] * math.cos(math.radians(ship2[-1]['heading'])),
                    head_width=5, head_length=5, color='g', zorder=2)

        # Plot the Gaussian model
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='Blues', alpha=0.5)
        if i == 1:
            plt.scatter(eCPA_points[:, 0], eCPA_points[:, 1], color='blue', marker='.', s=4)
            plt.scatter(eCPA_points[-7][0], eCPA_points[-7][1], color='green', marker='.', s=4)  # the linear eCPA
            plt.scatter(eCPA_points[-6:, 0], eCPA_points[-6:, 1], color='lawngreen', marker='.', s=4)  # the linear eCPA
        else:
            plt.scatter(eCPA_points[:, 0], eCPA_points[:, 1], color='blue', marker='.')
            if ozt_line is not None:
                plt.scatter(eCPA_points[-7][0], eCPA_points[-7][1], color='green', marker='.')  # the linear eCPA
                plt.scatter(eCPA_points[-6:, 0], eCPA_points[-6:, 1], color='lawngreen', marker='.')  # the linear eCPA
            else:
                plt.scatter(eCPA_points[-1][0], eCPA_points[-1][1], color='green', marker='.')
        plt.grid(True, alpha=0.5)

        # Draw a box around the maximum point in the first plot to indicate the zoom area in the second plot
        if i == 1:
            rect = plt.Rectangle((max_x - 20, max_y - 20), 40, 40, fill=False, color='black', linestyle='--', alpha=0.5)
            plt.gca().add_patch(rect)

        # Plot the maximum eCPA as red 'X'
        plt.plot(max_x, max_y, "x", markersize=11.5, markeredgewidth=4, color='white')
        plt.plot(max_x, max_y, "xr", markersize=10, markeredgewidth=2)

    # plt.show()

    plt.savefig("ecpa.png")
