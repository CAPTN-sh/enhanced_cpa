"""

A reusable enhanced version of the Closest Point of Approach (eCPA) for use in Gaussian models.

Author: Tom Beyer

"""

import math
import numpy as np
import matplotlib.pyplot as plt

from standard_cpa import calculate_standard_cpa
from dynamic_prediction import dynamic_prediction
from interpolate_path import interpolate_path
from real_dcpa import real_dcpa
from obstacle_zone_by_target import obstacle_zone_by_target

# from CPA.standard_cpa import calculate_standard_cpa
# from CPA.dynamic_prediction import dynamic_prediction
# from CPA.interpolate_path import interpolate_path
# from CPA.real_dcpa import real_dcpa
# from CPA.obstacle_zone_by_target import obstacle_zone_by_target


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


# Convert latitudes and longitudes to x and y coordinates (assuming a linear transformation in a local coordinate system)
def convert_coordinates(lat, lon):
    # Define the ranges for latitude, longitude, and the target coordinate system
    lat_range = (54.343316, 54.376374)
    lon_range = (10.135503, 10.199791)
    coord_range = (0, 300)

    # Compute the scale factors between the coordinate systems
    lat_scale = (coord_range[1] - coord_range[0]) / (lat_range[1] - lat_range[0])
    lon_scale = (coord_range[1] - coord_range[0]) / (lon_range[1] - lon_range[0])

    # Apply the scale factors to the latitude and longitude
    x = (lon - lon_range[0]) * lon_scale
    y = (lat - lat_range[0]) * lat_scale

    return x, y


# Compute the eCPA based on the given data.
def compute_ecpa(dynamic, 
                #  wind_enabled, 
                #  current_enabled, 
                 prediction_method, 
                 interpolation_method, 
                 forecast_steps, 
                 past_timesteps, 
                 r_dcpa, 
                 ozt, 
                 ship1, 
                 ship2, 
                #  wind, 
                #  current, 
                 security_radius=7):
    
    # # Adjust the ships SOGs and headings if wind data is included (only linear/non-dynamic mode)
    # if wind_enabled:
    #     # Convert the units
    #     wind_speed = wind[-1]['speed'] * 3.6 / 1.852  # m/s -> km/h -> knots
    #     wind_heading = (wind[-1]['direction'] + 180) % 360  # direction (from which the wind blows) -> heading
    #     # Add wind influence to the ships movements (respecting the given influence factor)
    #     ship1[-1]['SOG'], ship1[-1]['heading'] = add_vectors([ship1[-1]['SOG'], ship1[-1]['heading']], 
    #                                                 [wind_speed * ship1[-1]['wind_factor'], wind_heading])
    #     ship2[-1]['SOG'], ship2[-1]['heading'] = add_vectors([ship2[-1]['SOG'], ship2[-1]['heading']], 
    #                                                 [wind_speed * ship2[-1]['wind_factor'], wind_heading])

    # # Adjust the ships SOGs and headings if current data is included (only linear/non-dynamic mode)
    # if current_enabled:
    #     # Convert the units
    #     current_speed = current[-1]['speed'] * 0.036 / 1.852  # cm/s -> km/h -> knots
    #     current_heading = (current[-1]['direction'] + 180) % 360  # direction (from which the current flows) -> heading
    #     # Add current influence to the ships movements (respecting the given influence factor)
    #     ship1[-1]['SOG'], ship1[-1]['heading'] = add_vectors([ship1[-1]['SOG'], ship1[-1]['heading']], 
    #                                                 [current_speed * ship1[-1]['current_factor'], current_heading])
    #     ship2[-1]['SOG'], ship2[-1]['heading'] = add_vectors([ship2[-1]['SOG'], ship2[-1]['heading']], 
    #                                                 [current_speed * ship2[-1]['current_factor'], current_heading])
    


    # # Extract the position coordinates from the data
    # long1_positions = [entry['long'] for entry in ship1]
    # lat1_positions = [entry['lat'] for entry in ship1]
    # long2_positions = [entry['long'] for entry in ship2]
    # lat2_positions = [entry['lat'] for entry in ship2]

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

    # # Prepare the plot
    # plt.figure(figsize=(6, 6))
    # axis_measure = [0, 300, 0, 300]  # x-axis (start, finish), y-axis (start, finish)
    # plt.axis(axis_measure)
    # plt.grid(True)
    # img = plt.imread("../foerde_cutout.png")
    # plt.imshow(img, extent=axis_measure)

    # # Show wind info if enabled
    # if wind_enabled:
    #     # Plot the outer edge of the arrow
    #     plt.arrow(275, 25, wind_speed * math.sin(math.radians(wind_heading)),
    #               wind_speed * math.cos(math.radians(wind_heading)), 
    #               head_width=6, head_length=6, color='k', linewidth=3)
    #     # Plot the arrow itself
    #     plt.arrow(275, 25, wind_speed * math.sin(math.radians(wind_heading)),
    #               wind_speed * math.cos(math.radians(wind_heading)), 
    #               head_width=5, head_length=5, color='b', linewidth=2)
    #     # Plot some borders for clarity
    #     plt.plot([250, 250], [0, 50], color='k', linewidth=0.8)
    #     plt.plot([250, 300], [50, 50], color='k', linewidth=0.8)
    #     # Plot a label
    #     plt.text(264, 52, "Wind", fontsize=8, color='black')
        
    # # Show current info if enabled
    # if current_enabled:
    #     # Plot the outer edge of the arrow
    #     plt.arrow(225, 25, current_speed * math.sin(math.radians(current_heading)),
    #         current_speed * math.cos(math.radians(current_heading)), 
    #         head_width=6, head_length=6, color='k', linewidth=3)
    #     # Plot the arrow itself
    #     plt.arrow(225, 25, current_speed * math.sin(math.radians(current_heading)),
    #         current_speed * math.cos(math.radians(current_heading)), 
    #         head_width=5, head_length=5, color='w', linewidth=2)
    #     # Plot some borders for clarity
    #     plt.plot([200, 200], [0, 50], color='k', linewidth=0.8)
    #     plt.plot([200, 250], [50, 50], color='k', linewidth=0.8)
    #     plt.plot([250, 250], [0, 50], color='gray', linewidth=0.8)
    #     # Plot a label
    #     plt.text(208, 52, "Current", fontsize=8, color='black')

    # # Plot the ships
    # plt.plot(ship1[-1]['x'], ship1[-1]['y'], "ob", zorder=2)
    # plt.plot(ship2[-1]['x'], ship2[-1]['y'], "og", zorder=2)

    # # Plot the ships speed and heading as arrow
    # plt.arrow(ship1[-1]['x'], ship1[-1]['y'], ship1[-1]['SOG'] * math.sin(math.radians(ship1[-1]['heading'])),
    #             ship1[-1]['SOG'] * math.cos(math.radians(ship1[-1]['heading'])),
    #             head_width=5, head_length=5, color='b', zorder=2)
    # plt.arrow(ship2[-1]['x'], ship2[-1]['y'], ship2[-1]['SOG'] * math.sin(math.radians(ship2[-1]['heading'])),
    #             ship2[-1]['SOG'] * math.cos(math.radians(ship2[-1]['heading'])),
    #             head_width=5, head_length=5, color='g', zorder=2)
    
    # # Plot the past positions of the ships as well as their speed and course 
    # for i in range(len(ship1) - 1):
    #     plt.plot(ship1[i]['x'], ship1[i]['y'], 'ob', mfc='none', zorder=2)
    #     plt.arrow(ship1[i]['x'], ship1[i]['y'], ship1[-1]['SOG'] * math.sin(math.radians(ship1[i]['heading'])),
    #                 ship1[i]['SOG'] * math.cos(math.radians(ship1[i]['heading'])),
    #                 head_width=5, head_length=5, color='b', zorder=2)
    # for i in range(len(ship2) - 1):
    #     plt.plot(ship2[i]['x'], ship2[i]['y'], 'og', mfc='none', zorder=2)
    #     plt.arrow(ship2[i]['x'], ship2[i]['y'], ship2[-1]['SOG'] * math.sin(math.radians(ship2[i]['heading'])),
    #         ship2[i]['SOG'] * math.cos(math.radians(ship2[i]['heading'])),
    #         head_width=5, head_length=5, color='g', zorder=2)

    # # If dynamic courses are activated plot the respective predicted positions
    # if dynamic:
    #     for i in range(forecast_steps):
    #         # Ship 1
    #         plt.plot(predict_position1[i][0], predict_position1[i][1], 'o', color='darkblue', mfc='none', zorder=2)
    #         plt.arrow(predict_position1[i][0], predict_position1[i][1], predict_speed1[i] * math.sin(math.radians(predict_course1[i])),
    #                     predict_speed1[i] * math.cos(math.radians(predict_course1[i])),
    #                     head_width=5, head_length=5, color='darkblue', zorder=2)
    #         # Ship 2
    #         plt.plot(predict_position2[i][0], predict_position2[i][1], 'o', color='darkgreen', mfc='none', zorder=2)
    #         plt.arrow(predict_position2[i][0], predict_position2[i][1], predict_speed2[i] * math.sin(math.radians(predict_course2[i])),
    #                     predict_speed2[i] * math.cos(math.radians(predict_course2[i])),
    #                     head_width=5, head_length=5, color='darkgreen', zorder=2)
        
    # # Plot the interpolated course of the ships
    # if dynamic:
    #     # Plot the interpolated paths
    #     plt.plot([ele[0] for ele in interpolated_predict_xy1], [ele[1] for ele in interpolated_predict_xy1], color='darkblue', linestyle='--', linewidth=0.8)
    #     plt.plot([ele[0] for ele in interpolated_predict_xy2], [ele[1] for ele in interpolated_predict_xy2], color='darkgreen', linestyle='--', linewidth=0.8)
    #     plt.plot([ele[0] for ele in short_interpolated_xy1], [ele[1] for ele in short_interpolated_xy1], color='b', linewidth=0.8)
    #     plt.plot([ele[0] for ele in short_interpolated_xy2], [ele[1] for ele in short_interpolated_xy2], color='g', linewidth=0.8)

    # # Otherwise plot lines and project the actual course linearly
    # else:
    #     # Plot linear course interpolations
    #     plt.plot(x1_positions, y1_positions, linewidth=0.8, color='b')
    #     plt.plot(x2_positions, y2_positions, linewidth=0.8, color='g')
    #     # Predict future courses linearly
    #     x1_heading = [ship1[-1]['x'], ship1[-1]['x'] + math.sin(math.radians(ship1[-1]['heading'])) * axis_measure[1]]
    #     y1_heading = [ship1[-1]['y'], ship1[-1]['y'] + math.cos(math.radians(ship1[-1]['heading'])) * axis_measure[3]]
    #     x2_heading = [ship2[-1]['x'], ship2[-1]['x'] + math.sin(math.radians(ship2[-1]['heading'])) * axis_measure[1]]
    #     y2_heading = [ship2[-1]['y'], ship2[-1]['y'] + math.cos(math.radians(ship2[-1]['heading'])) * axis_measure[3]]
    #     plt.plot(x1_heading, y1_heading, color='darkblue', linestyle='--', linewidth=0.8, zorder=1)
    #     plt.plot(x2_heading, y2_heading, color='darkgreen', linestyle='--', linewidth=0.8, zorder=1)
    #     # If activated also plot the OZT line
    #     if ozt:
    #         plt.plot([ozt_points[0][0], ozt_points[1][0]], [ozt_points[0][1], ozt_points[1][1]], marker='X', color='lawngreen')
    
    # # Plot the CPA as red 'X' and print the relevant numbers
    # plt.plot(final_cpa[0][0], final_cpa[0][1], "x", markersize=11.5, markeredgewidth=4, color='white')
    # plt.plot(final_cpa[0][0], final_cpa[0][1], "xr", markersize=10, markeredgewidth=2)
    
    # Print the relevant numbers
    # print(f"Closest Point of Approach (CPA): (({final_cpa[0][0]}, {final_cpa[0][1]}), ({final_cpa[1][0]}, {final_cpa[1][1]}))")
    # print(f"Distance at CPA (DCPA): {cpa_distance}")
    # if r_dcpa:
    #     print(f"Real Distance at CPA (rDCPA): {real_cpa_distance}")
    # print(f"Time to CPA (TCPA): {time_to_cpa}")
    # if not dynamic and ozt:
    #     print(f"Obstacle Zone to Target (OZT): {ozt_points}")

    # plt.show()
    
    # Return the results
    if ozt:
        return ozt_points, time_to_cpa, cpa_distance, real_cpa_distance
    else:
        return final_cpa[0], time_to_cpa, cpa_distance, real_cpa_distance
