"""

Interpolate a given two-dimensional path by a selected method.

Author: Tom Beyer

"""    

import numpy as np

from scipy.interpolate import splprep, splev, interp1d, CubicSpline, CubicHermiteSpline, PchipInterpolator, Akima1DInterpolator

# Function to check if the first four values given in a list with a while loop as long as some of them are equal 
# to each other and if, vary them by a small amount.
def check_values(values):
    count = 0
    while (values[0] == values[1] or values[0] == values[2] or values[0] == values[3] or 
           values[1] == values[2] or values[1] == values[3] or values[2] == values[3]) and count < 1000:
        for i in range(3):
            if values[i] == values[i + 1]:
                values[i] += 1e-8 * (i + 1)
        count += 1

    return values


# A function that adds a very small amount to every value in a list (based on its index) so that they differ.
def add_small_amount(lst):
    for i in range(len(lst)):
        lst[i] += 0.00000001 * i

    return lst

# Interpolate a given path by the chosen method.
def interpolate_path(x, y, interpolation_method, speed_x=[], speed_y=[]):
    # Make sure the data is in the right format
    x = np.asarray(x)
    y = np.asarray(y)
    speed_x = np.asarray(speed_x)
    speed_y = np.asarray(speed_y)
    # Initialize resulting paths
    interpolated_x = []
    interpolated_y = []
    # Compute the interpolation by making use of parameterization of the path-curves
    # (otherwise a not strictly increasing x would pose a problem)
    if interpolation_method == 'linear':
        # Create parameter values based on the number of data points
        t = np.linspace(0, 1, len(x))
        # Create interpolation functions for x and y separately
        interpolator_x = interp1d(t, x)
        interpolator_y = interp1d(t, y)
        # Define a set of parameter values for the interpolated curve
        u_values = np.linspace(0, 1, 1000)
        # Interpolate x and y coordinates separately
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
    if interpolation_method == 'quadratic':
        # Create parameter values based on the number of data points
        t = np.linspace(0, 1, len(x))
        # Create interpolation functions for x and y separately
        interpolator_x = interp1d(t, x, kind='quadratic')
        interpolator_y = interp1d(t, y, kind='quadratic')
        # Define a set of parameter values for the interpolated curve
        u_values = np.linspace(0, 1, 1000)
        # Interpolate x and y coordinates separately
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
    elif interpolation_method == 'cubic':
        # Create parameter values based on the number of data points
        t = np.linspace(0, 1, len(x))
        # Create interpolation functions for x and y separately
        interpolator_x = interp1d(t, x, kind='cubic')
        interpolator_y = interp1d(t, y, kind='cubic')
        # Define a set of parameter values for the interpolated curve
        u_values = np.linspace(0, 1, 1000)
        # Interpolate x and y coordinates separately
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
    elif interpolation_method == 'cubic_spline':
        # Prepare the data with parameterization
        tck, _ = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        # Create a CubicSpline object using the interpolated data
        cs = CubicSpline(u_new, np.column_stack((x_new, y_new)))
        # Evaluate the CubicSpline at specific parameter values
        u_values = np.linspace(0, 1, 1000)
        curve_points = cs(u_values)
        # Extract the interpolated paths data
        interpolated_x = curve_points[:, 0]
        interpolated_y = curve_points[:, 1]
    elif interpolation_method == 'cubic_hermite_spline':
        # Prepare the data with parameterization
        tck, _ = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        # Make sure at least four values are different
        speed_x = add_small_amount(speed_x)
        speed_y = add_small_amount(speed_y)
        tck, _ = splprep([speed_x, speed_y], s=0)
        speed_x_new, speed_y_new = splev(u_new, tck)
        # Create a CubicSpline object using the interpolated data
        cs = CubicHermiteSpline(u_new, np.column_stack((x_new, y_new)), np.column_stack((speed_x_new, speed_y_new)))
        # Evaluate the CubicSpline at specific parameter values
        u_values = np.linspace(0, 1, 1000)
        curve_points = cs(u_values)
        # Extract the interpolated paths data
        interpolated_x = curve_points[:, 0]
        interpolated_y = curve_points[:, 1]
    elif interpolation_method == 'pchip':
        # Prepare the data with parameterization
        tck, _ = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        # Create a CubicSpline object using the interpolated data
        pchip_interp = PchipInterpolator(u_new, np.column_stack((x_new, y_new)))
        # Evaluate the CubicSpline at specific parameter values
        u_values = np.linspace(0, 1, 1000)
        curve_points = pchip_interp(u_values)
        # Extract the interpolated paths data
        interpolated_x = curve_points[:, 0]
        interpolated_y = curve_points[:, 1]
    elif interpolation_method == 'akima':
        # Prepare the data with parameterization
        tck, _ = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_new, tck)
        # Create a CubicSpline object using the interpolated data
        akima_interp = Akima1DInterpolator(u_new, np.column_stack((x_new, y_new)))
        # Evaluate the CubicSpline at specific parameter values
        u_values = np.linspace(0, 1, 1000)
        curve_points = akima_interp(u_values)
        # Extract the interpolated paths data
        interpolated_x = curve_points[:, 0]
        interpolated_y = curve_points[:, 1]
    
    return interpolated_x, interpolated_y