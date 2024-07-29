import math
import numpy as np

def add_vectors(vector1, vector2):
    magnitude1, angle1_degrees = vector1
    magnitude2, angle2_degrees = vector2
    x1 = magnitude1 * math.cos(math.radians(angle1_degrees))
    y1 = magnitude1 * math.sin(math.radians(angle1_degrees))
    x2 = magnitude2 * math.cos(math.radians(angle2_degrees))
    y2 = magnitude2 * math.sin(math.radians(angle2_degrees))
    result_x = x1 + x2
    result_y = y1 + y2
    result_magnitude = math.sqrt(result_x ** 2 + result_y ** 2)
    result_angle = (math.degrees(math.atan2(result_y, result_x)) + 360) % 360
    return result_magnitude, result_angle

def find_closest_index(number, array):
    min_time_difference = float('inf')
    closest_time_index = None
    for i in range(len(array)):
        temp_time_difference = abs(array[i] - number)
        if temp_time_difference < min_time_difference:
            min_time_difference = temp_time_difference
            closest_time_index = i
    return closest_time_index

def interpolate_timestamps(original_waypoints, interpolated_waypoints, original_timestamps):
    original_waypoints = np.asarray(original_waypoints)
    interpolated_waypoints = np.asarray(interpolated_waypoints)
    indices = []
    interpolated_index = 0
    for waypoint in original_waypoints:
        min_distance = float('inf')
        min_index = interpolated_index
        for i in range(interpolated_index, len(interpolated_waypoints)):
            distance = np.linalg.norm(waypoint - interpolated_waypoints[i])
            if distance < min_distance:
                min_distance = distance
                min_index = i
        interpolated_index = min_index
        indices.append(min_index)
    interpolated_timestamps = []
    for i in range(len(original_waypoints) - 1):
        temp_interpolated_time = np.linspace(original_timestamps[i], original_timestamps[i + 1], (indices[i + 1] - indices[i]), endpoint=False)
        interpolated_timestamps += list(temp_interpolated_time)
    interpolated_timestamps.append(original_timestamps[-1])
    return interpolated_timestamps

def clockwise_angle(x, y):
    angle_rad = math.atan2(x, y)
    angle_deg = math.degrees(angle_rad)
    clockwise_angle = (angle_deg + 360) % 360
    return clockwise_angle