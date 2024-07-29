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




def interpolate_path_scenerios(x, y, timestamps, u,method='linear'):
    x = np.asarray(x)
    y = np.asarray(y)
    timestamps = np.asarray(timestamps)
    t = np.linspace(0, 1, len(x))
    u_values = np.linspace(0, 1, u)

    if method == 'linear':
        interpolator_x = interp1d(t, x)
        interpolator_y = interp1d(t, y)
        interpolator_timestamps = interp1d(t, timestamps)
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
        interpolated_timestamps = interpolator_timestamps(u_values)
    elif method == 'quadratic':
        interpolator_x = interp1d(t, x, kind='quadratic')
        interpolator_y = interp1d(t, y, kind='quadratic')
        interpolator_timestamps = interp1d(t, timestamps, kind='quadratic')
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
        interpolated_timestamps = interpolator_timestamps(u_values)
    elif method == 'cubic':
        interpolator_x = interp1d(t, x, kind='cubic')
        interpolator_y = interp1d(t, y, kind='cubic')
        interpolator_timestamps = interp1d(t, timestamps, kind='cubic')
        interpolated_x = interpolator_x(u_values)
        interpolated_y = interpolator_y(u_values)
        interpolated_timestamps = interpolator_timestamps(u_values)
    elif method == 'cubic_spline':
        tck, _ = splprep([x, y], s=0)
        x_new, y_new = splev(u_values, tck)
        interpolator_timestamps = interp1d(t, timestamps, kind='cubic')
        return x_new, y_new, interpolator_timestamps(u_values)
    elif method == 'cubic_hermite_spline':
        tck, _ = splprep([x, y], s=0)
        x_new, y_new = splev(u_values, tck)
        speed_x = add_small_amount(np.diff(x_new, prepend=x_new[0]))
        speed_y = add_small_amount(np.diff(y_new, prepend=y_new[0]))
        cs = CubicHermiteSpline(u_values, np.column_stack((x_new, y_new)), np.column_stack((speed_x, speed_y)))
        curve_points = cs(u_values)
        interpolator_timestamps = interp1d(t, timestamps, kind='cubic')
        return curve_points[:, 0], curve_points[:, 1], interpolator_timestamps(u_values)
    elif method == 'pchip':
        pchip_interp = PchipInterpolator(t, np.column_stack((x, y)))
        curve_points = pchip_interp(u_values)
        interpolator_timestamps = interp1d(t, timestamps, kind='cubic')
        return curve_points[:, 0], curve_points[:, 1], interpolator_timestamps(u_values)
    elif method == 'akima':
        akima_interp = Akima1DInterpolator(t, np.column_stack((x, y)))
        curve_points = akima_interp(u_values)
        interpolator_timestamps = interp1d(t, timestamps, kind='cubic')
        return curve_points[:, 0], curve_points[:, 1], interpolator_timestamps(u_values)

    interpolated_x = interpolator_x(u_values)
    interpolated_y = interpolator_y(u_values)
    interpolated_timestamps = interpolator_timestamps(u_values)
    return interpolated_x, interpolated_y, interpolated_timestamps


def interpolate_scenarios(scenarios, interpolation_method='linear', u=100):
    all_data = {}

    for scenario_name, scenario_data in scenarios.items():
        own_ship_data = scenario_data['own_ship']
        target_ship_data = scenario_data['target_ship']

        # Extract features for own ship
        own_ship_features = {key: [point[key] for point in own_ship_data] for key in own_ship_data[0].keys()}
        timestamps = np.linspace(min(own_ship_features['timestamp']), max(own_ship_features['timestamp']), u)

        target_ship_features = {key: [point[key] for point in target_ship_data] for key in target_ship_data[0].keys()}
        target_timestamps = np.linspace(min(target_ship_features['timestamp']), max(target_ship_features['timestamp']), u)

        if interpolation_method == 'cubic_spline':
            interpolator = CubicSpline
        elif interpolation_method == 'pchip':
            interpolator = PchipInterpolator
        elif interpolation_method == 'akima':
            interpolator = Akima1DInterpolator
        elif interpolation_method == 'cubic_hermite_spline':
            interpolated_own_ship_features = {
                key: CubicHermiteSpline(own_ship_features['timestamp'], own_ship_features[key], np.gradient(own_ship_features[key], own_ship_features['timestamp']))(timestamps)
                for key in own_ship_features.keys()
            }
            interpolated_target_ship_features = {
                key: CubicHermiteSpline(target_ship_features['timestamp'], target_ship_features[key], np.gradient(target_ship_features[key], target_ship_features['timestamp']))(target_timestamps)
                for key in target_ship_features.keys()
            }
        else:
            interpolator = interp1d

        if interpolation_method not in ['akima', 'cubic_hermite_spline']:
            if interpolation_method in ['linear', 'quadratic', 'cubic']:
                interpolated_own_ship_features = {
                    key: interpolator(own_ship_features['timestamp'], own_ship_features[key], kind=interpolation_method)(timestamps)
                    for key in own_ship_features.keys()
                }
            elif interpolation_method in ['cubic_spline', 'pchip']:
                interpolated_own_ship_features = {
                    key: interpolator(own_ship_features['timestamp'], own_ship_features[key])(timestamps)
                    for key in own_ship_features.keys()
                }
            else:
                raise NotImplementedError(f"{interpolation_method} is unsupported by available interpolation methods.")

            if interpolation_method in ['linear', 'quadratic', 'cubic']:
                interpolated_target_ship_features = {
                    key: interpolator(target_ship_features['timestamp'], target_ship_features[key], kind=interpolation_method)(target_timestamps)
                    for key in target_ship_features.keys()
                }
            elif interpolation_method in ['cubic_spline', 'pchip']:
                interpolated_target_ship_features = {
                    key: interpolator(target_ship_features['timestamp'], target_ship_features[key])(target_timestamps)
                    for key in target_ship_features.keys()
                }
            else:
                raise NotImplementedError(f"{interpolation_method} is unsupported by available interpolation methods.")
        elif interpolation_method == 'akima':
            interpolated_own_ship_features = {
                key: Akima1DInterpolator(own_ship_features['timestamp'], own_ship_features[key])(timestamps)
                for key in own_ship_features.keys()
            }
            interpolated_target_ship_features = {
                key: Akima1DInterpolator(target_ship_features['timestamp'], target_ship_features[key])(target_timestamps)
                for key in target_ship_features.keys()
            }

        # Combine interpolated features into dictionaries
        interpolated_own_ship_data = [{key: interpolated_own_ship_features[key][i] for key in interpolated_own_ship_features.keys()} for i in range(u)]
        interpolated_target_ship_data = [{key: interpolated_target_ship_features[key][i] for key in interpolated_target_ship_features.keys()} for i in range(u)]

        final_cpa = scenario_data['final_cpa']
        final_tcpa = scenario_data.get('final_tcpa', None)
        final_dcpa = scenario_data.get('final_dcpa', None)

        all_data[scenario_name] = {
            'own_ship': interpolated_own_ship_data,
            'target_ship': interpolated_target_ship_data,
            'final_cpa': final_cpa,
            'final_tcpa': final_tcpa,
            'final_dcpa': final_dcpa
        }
    return all_data

# def interpolate_scenarios(scenarios, interpolation_method='linear', u=100):
#     all_data = {}

#     for scenario_name, scenario_data in scenarios.items():
#         own_ship_data = scenario_data['own_ship']
#         target_ship_data = scenario_data['target_ship']


#         own_ship_x = [point['x'] for point in own_ship_data]
#         own_ship_y = [point['y'] for point in own_ship_data]
#         own_ship_timestamps = [point['timestamp'] for point in own_ship_data]
#         target_ship_x = [point['x'] for point in target_ship_data]
#         target_ship_y = [point['y'] for point in target_ship_data]
#         target_ship_timestamps = [point['timestamp'] for point in target_ship_data]

#         interpolated_own_ship_x, interpolated_own_ship_y, interpolated_own_ship_timestamps = interpolate_path_scenerios(own_ship_x, own_ship_y, own_ship_timestamps, u, interpolation_method)
#         interpolated_target_ship_x, interpolated_target_ship_y, interpolated_target_ship_timestamps = interpolate_path_scenerios(target_ship_x, target_ship_y, target_ship_timestamps, u, interpolation_method)
        
#         # print(f"Scenario: {scenario_name}")
#         # print(f"Original own_ship data points: {len(own_ship_data)}")
#         # print(f"Interpolated own_ship data points: {len(interpolated_own_ship_x)}")
#         # print(f"Original target_ship data points: {len(target_ship_data)}")
#         # print(f"Interpolated target_ship data points: {len(interpolated_target_ship_x)}\n")

#         own_ship_data = [{'x': x, 'y': y, 'COG': own_ship_data[0]['COG'], 'SOG': own_ship_data[0]['SOG'],
#                             'heading': own_ship_data[0]['heading'], 'timestamp': t, 'wind_factor': own_ship_data[0]['wind_factor'],
#                             'current_factor': own_ship_data[0]['current_factor'], 'bow': own_ship_data[0]['bow'],
#                             'stern': own_ship_data[0]['stern'], 'portside': own_ship_data[0]['portside'],
#                             'starboard': own_ship_data[0]['starboard']} for x, y, t in zip(interpolated_own_ship_x, interpolated_own_ship_y, interpolated_own_ship_timestamps)]
        
#         target_ship_data = [{'x': x, 'y': y, 'COG': target_ship_data[0]['COG'], 'SOG': target_ship_data[0]['SOG'],
#                                 'heading': target_ship_data[0]['heading'], 'timestamp': t, 'wind_factor': target_ship_data[0]['wind_factor'],
#                                 'current_factor': target_ship_data[0]['current_factor'], 'bow': target_ship_data[0]['bow'],
#                                 'stern': target_ship_data[0]['stern'], 'portside': target_ship_data[0]['portside'],
#                                 'starboard': target_ship_data[0]['starboard']} for x, y, t in zip(interpolated_target_ship_x, interpolated_target_ship_y, interpolated_target_ship_timestamps)]
            
#         final_cpa = scenario_data['final_cpa']
#         final_tcpa = scenario_data.get('final_tcpa', None)
#         final_dcpa = scenario_data.get('final_dcpa', None)

#         all_data[scenario_name] = {
#             'own_ship': own_ship_data,
#             'target_ship': target_ship_data,
#             'final_cpa': final_cpa,
#             'final_tcpa': final_tcpa,
#             'final_dcpa': final_dcpa
#         }
#     return all_data