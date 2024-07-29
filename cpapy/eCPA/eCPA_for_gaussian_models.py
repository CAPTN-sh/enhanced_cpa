"""

A reusable enhanced version of the Closest Point of Approach (eCPA) for use in Gaussian models.

Author: Tom Beyer

"""

import math
import numpy as np

from standard_cpa import calculate_standard_cpa
from dynamic_prediction import dynamic_prediction
from interpolate_path import interpolate_path
from real_dcpa import real_dcpa
from obstacle_zone_by_target import obstacle_zone_by_target


"""
Functions
"""

# Add two vectors that are just given by their magnitudes and directions.
# Returns also a vector with just a magnitude and a direction.
def add_vectors(vector1, vector2):
    # Extract values
    magnitude1, angle1_degrees = vector1
    magnitude2, angle2_degrees = vector2
    # Calculate x and y components for both vectors
    x1 = magnitude1 * math.cos(math.radians(angle1_degrees))
    y1 = magnitude1 * math.sin(math.radians(angle1_degrees))
    x2 = magnitude2 * math.cos(math.radians(angle2_degrees))
    y2 = magnitude2 * math.sin(math.radians(angle2_degrees))
    # Add the x and y components to find the resultant vector
    result_x = x1 + x2
    result_y = y1 + y2
    # Calculate the magnitude (length) of the resultant vector
    result_magnitude = math.sqrt(result_x ** 2 + result_y ** 2)
    # Calculate the angle of the resultant vector in degrees and adjust for the coordinate system
    result_angle = (math.degrees(math.atan2(result_y, result_x)) + 360) % 360

    return result_magnitude, result_angle


# Find the closest array entry to a given number.
def find_closest_index(number, array):
    min_time_difference = float('inf')
    closest_time_index = None
    for i in range(len(array)):
        temp_time_difference = abs(array[i] - number)
        if temp_time_difference < min_time_difference:
            min_time_difference = temp_time_difference
            closest_time_index = i

    return closest_time_index


# Built a list of interpolated timestamps that correspond closely to some interpolated path.
def interpolate_timestamps(original_waypoints, interpolated_waypoints, original_timestamps):
    # Format the data
    original_waypoints = np.asarray(original_waypoints)
    interpolated_waypoints = np.asarray(interpolated_waypoints)
    # Initialize an array to store the indices of corresponding points in the interpolated array
    indices = []
    interpolated_index = 0
    # Loop through each waypoint
    for waypoint in original_waypoints:
        # Initialize the minimum distance and index
        min_distance = float('inf')
        min_index = interpolated_index
        # Iterate through the remaining points in the interpolated array
        for i in range(interpolated_index, len(interpolated_waypoints)):
            distance = np.linalg.norm(waypoint - interpolated_waypoints[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        # Update the index in the interpolated array
        interpolated_index = min_index
        # Add the index to the indices array
        indices.append(min_index)

    # Built the list of corresponding timestamps based on the found indices
    # Initialize an array for the result
    interpolated_timestamps = []
    for i in range(len(original_waypoints) - 1):
        temp_interpolated_time = np.linspace(original_timestamps[i], original_timestamps[i + 1], (indices[i + 1] - indices[i]), endpoint=False)
        interpolated_timestamps += list(temp_interpolated_time)
    interpolated_timestamps.append(original_timestamps[-1])

    return interpolated_timestamps


# Calculate the clockwise angle of a 2D vector in a coordinate system.
def clockwise_angle(x, y):
    # Calculate the angle in radians
    angle_rad = math.atan2(x, y)
    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)
    # Adjust the angle to be in the 360° range
    clockwise_angle = (angle_deg + 360) % 360

    return clockwise_angle


# Compute the eCPA based on the given data.
def compute_ecpa(dynamic, 
                 wind_enabled, 
                 current_enabled, 
                 prediction_method, 
                 interpolation_method, 
                 forecast_steps, 
                 past_timesteps, 
                 r_dcpa, 
                 ozt, 
                 ship1, 
                 ship2, 
                 wind, 
                 current, 
                 security_radius=7):
    
    # Adjust the ships SOGs and headings if wind data is included (only linear/non-dynamic mode)
    if wind_enabled:
        # Convert the units
        wind_speed = wind[-1]['speed'] * 3.6 / 1.852  # m/s -> km/h -> knots
        wind_heading = (wind[-1]['direction'] + 180) % 360  # direction (from which the wind blows) -> heading
        # Add wind influence to the ships movements (respecting the given influence factor)
        ship1[-1]['SOG'], ship1[-1]['heading'] = add_vectors([ship1[-1]['SOG'], ship1[-1]['heading']], 
                                                    [wind_speed * ship1[-1]['wind_factor'], wind_heading])
        ship2[-1]['SOG'], ship2[-1]['heading'] = add_vectors([ship2[-1]['SOG'], ship2[-1]['heading']], 
                                                    [wind_speed * ship2[-1]['wind_factor'], wind_heading])

    # Adjust the ships SOGs and headings if current data is included (only linear/non-dynamic mode)
    if current_enabled:
        # Convert the units
        current_speed = current[-1]['speed'] * 0.036 / 1.852  # cm/s -> km/h -> knots
        current_heading = (current[-1]['direction'] + 180) % 360  # direction (from which the current flows) -> heading
        # Add current influence to the ships movements (respecting the given influence factor)
        ship1[-1]['SOG'], ship1[-1]['heading'] = add_vectors([ship1[-1]['SOG'], ship1[-1]['heading']], 
                                                    [current_speed * ship1[-1]['current_factor'], current_heading])
        ship2[-1]['SOG'], ship2[-1]['heading'] = add_vectors([ship2[-1]['SOG'], ship2[-1]['heading']], 
                                                    [current_speed * ship2[-1]['current_factor'], current_heading])
    
    # Extract the position coordinates from the data
    x1_positions = [entry['x'] for entry in ship1]
    y1_positions = [entry['y'] for entry in ship1]
    x2_positions = [entry['x'] for entry in ship2]
    y2_positions = [entry['y'] for entry in ship2]

    # Make a prediction for future positions if dynamic mode (non-linear courses) is enabled
    if dynamic:
        # Ship 1
        predict_position1, predict_course1, predict_speed1, predict_time1 = dynamic_prediction(ship1, past_timesteps, forecast_steps, prediction_method)
        # Ship 2
        predict_position2, predict_course2, predict_speed2, predict_time2 = dynamic_prediction(ship2, past_timesteps, forecast_steps, prediction_method)

    # If dynamic courses are activated interpolate by the chosen method
    if dynamic:
        # Add predictions to the position data
        for i in range(forecast_steps):
            x1_positions.append(predict_position1[i][0])
            y1_positions.append(predict_position1[i][1])
            x2_positions.append(predict_position2[i][0])
            y2_positions.append(predict_position2[i][1])
        if interpolation_method == 'cubic_hermite_spline':
            # Compute the x and y components of the ships movement (needed for cubic_hermite_spline)
            # Initialize lists with the resulting components
            speed_x1 = []
            speed_y1 = []
            speed_x2 = []
            speed_y2 = []
            # Collect the data
            speed1 = [entry['SOG'] for entry in ship1] + list(predict_speed1)
            speed2 = [entry['SOG'] for entry in ship2] + list(predict_speed2)
            course1 = [entry['COG'] for entry in ship1] + list(predict_course1)
            course2 = [entry['COG'] for entry in ship2] + list(predict_course2)
            # Determine the components of the course of ship1
            for i in range(len(speed1)):
                speed_x1.append(speed1[i] * math.sin(math.radians(course1[i])))
                speed_y1.append(speed1[i] * math.cos(math.radians(course1[i])))
            # Determine the components of the course of ship2
            for i in range(len(speed2)):
                speed_x2.append(speed2[i] * math.sin(math.radians(course2[i])))
                speed_y2.append(speed2[i] * math.cos(math.radians(course2[i])))
            # Interpolate the paths by Cubic Hermit Spline
            interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method, speed_x1, speed_y1)
            interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method, speed_x2, speed_y2)
        else:
            # Interpolate the paths by the chosen method
            interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method)
            interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method)

    # Compute the CPA
    # If dynamic courses are active
    if dynamic:
        # Collect and format the necessary data
        interpolated_xy1 = list(zip(interpolated_x1, interpolated_y1))
        interpolated_xy2 = list(zip(interpolated_x2, interpolated_y2))
        xy1_positions = list(zip(x1_positions, y1_positions))
        xy2_positions = list(zip(x2_positions, y2_positions))
        timestamps1 = [entry['timestamp'] for entry in ship1] + predict_time1
        timestamps2 = [entry['timestamp'] for entry in ship2] + predict_time2
        # Built a list of timestamps that correspond to the interpolated paths
        interpolated_timestamps1 = interpolate_timestamps(xy1_positions, interpolated_xy1, timestamps1)
        interpolated_timestamps2 = interpolate_timestamps(xy2_positions, interpolated_xy2, timestamps2)
        # Slice the lists from the last real position of own ship (ship1) to the rest of the predictions
        closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps1)  # for ship1
        interpolated_predict_xy1 = interpolated_xy1[closest_timestamp_index:]  # the predicted positions
        short_interpolated_xy1 = interpolated_xy1[:closest_timestamp_index]  # the old positions
        interpolated_predict_timestamps1 = interpolated_timestamps1[closest_timestamp_index:]
        closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps2)  # for ship2
        interpolated_predict_xy2 = interpolated_xy2[closest_timestamp_index:]  # the predicted positions
        short_interpolated_xy2 = interpolated_xy2[:closest_timestamp_index]  # the old positions
        interpolated_predict_timestamps2 = interpolated_timestamps2[closest_timestamp_index:]
        # Built a list with all synchronized distances
        distances = []  # result list
        for i in range(len(interpolated_predict_xy1)):
            # Find the closest corresponding interpolated timestamp for ship2
            closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[i], interpolated_predict_timestamps2)
            # Calculate the Euclidean distance between the respective interpolated positions
            distance = np.linalg.norm(np.asarray(interpolated_predict_xy1[i]) - np.asarray(interpolated_predict_xy2[closest_timestamp_index]))
            # Save result
            distances.append(distance)
        # Find the minimum distance in the resulting list
        min_distance_index = distances.index(min(distances))
        # Determine the corresponding position of ship1
        cpa_position1 = interpolated_predict_xy1[min_distance_index]
        # Determine the corresponding position of ship2
        closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[min_distance_index], interpolated_predict_timestamps2)
        cpa_position2 = interpolated_predict_xy2[closest_timestamp_index]
        # Built the final CPA values
        final_cpa = (cpa_position1, cpa_position2)
        cpa_distance = min(distances)
        time_to_cpa = interpolated_predict_timestamps1[min_distance_index] - ship1[-1]['timestamp']

        # Save positions before and/or after CPA for the real DCPA computation
        if r_dcpa:
            # Initialize variables
            before_cpa_position1 = None
            before_cpa_position2 = None
            after_cpa_position1 = None
            after_cpa_position2 = None
            # Determine which positions are available
            if min_distance_index > 0 and closest_timestamp_index > 0:
                # Save positions before CPA
                before_cpa_position1 = interpolated_predict_xy1[min_distance_index - 1]
                before_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index - 1]
            else:
                # Save positions after CPA
                after_cpa_position1 = interpolated_predict_xy1[min_distance_index + 1]
                after_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index + 1]

    else:
        # Otherwise compute the standard linear CPA
        final_cpa, cpa_distance, time_to_cpa = calculate_standard_cpa(ship1[-1], ship2[-1])
        # And the OZT if this option is chosen
        if ozt:
            ozt_points = obstacle_zone_by_target(security_radius, (ship1[-1]['x'], ship1[-1]['y']), (ship2[-1]['x'], ship2[-1]['y']), 
                                                ship2[-1]['heading'], ship1[-1]['SOG'], ship2[-1]['SOG'])

    # Compute the approximate real DCPA if this option is chosen
    real_cpa_distance = None  # Initialize in any case for the return structure
    if r_dcpa:
        # Initialize variables
        heading1 = None
        heading2 = None
        # The dynamic case
        if dynamic:
            # Determine the approximate heading of both vessels at the CPA
            if after_cpa_position1 != None:
                heading1 = clockwise_angle(after_cpa_position1[0] - cpa_position1[0], after_cpa_position1[1] - cpa_position1[1])
                heading2 = clockwise_angle(after_cpa_position2[0] - cpa_position2[0], after_cpa_position2[1] - cpa_position2[1])
            else:
                heading1 = clockwise_angle(cpa_position1[0] - before_cpa_position1[0], cpa_position1[1] - before_cpa_position1[1])
                heading2 = clockwise_angle(cpa_position2[0] - before_cpa_position2[0], cpa_position2[1] - before_cpa_position2[1])
            # Determine the real DCPA
            real_cpa_distance = real_dcpa(final_cpa, ship1[-1], heading1, ship2[-1], heading2)
        # The linear case
        else:
            # Determine the approximate heading of both vessels at the CPA
            heading1 = ship1[-1]['heading']
            heading2 = ship2[-1]['heading']
            # Determine the real DCPA
            real_cpa_distance = real_dcpa(final_cpa, ship1[-1], heading1, ship2[-1], heading2)
    
    # Return the results
    if ozt:
        return ozt_points, time_to_cpa, cpa_distance, real_cpa_distance
    else:
        return final_cpa[0], time_to_cpa, cpa_distance, real_cpa_distance

