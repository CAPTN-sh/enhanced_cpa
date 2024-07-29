import math
import numpy as np

from standard_cpa import calculate_standard_cpa
from dynamic_prediction import dynamic_prediction
from utils_ import add_vectors, find_closest_index, interpolate_timestamps, clockwise_angle
from interpolate_path import interpolate_path
from real_dcpa import real_dcpa
from obstacle_zone_by_target import obstacle_zone_by_target
from typing import List, Tuple, Dict, Callable

def adjust_for_wind_and_current(ship, wind, current, wind_enabled, current_enabled):
    if wind_enabled:
        wind_speed = wind[-1]['speed'] * 3.6 / 1.852
        wind_heading = (wind[-1]['direction'] + 180) % 360
        ship[-1]['SOG'], ship[-1]['heading'] = add_vectors([ship[-1]['SOG'], ship[-1]['heading']], 
                                                           [wind_speed * ship[-1]['wind_factor'], wind_heading])
    if current_enabled:
        current_speed = current[-1]['speed'] * 0.036 / 1.852
        current_heading = (current[-1]['direction'] + 180) % 360
        ship[-1]['SOG'], ship[-1]['heading'] = add_vectors([ship[-1]['SOG'], ship[-1]['heading']], 
                                                           [current_speed * ship[-1]['current_factor'], current_heading])
    return ship

def calculate_dynamic_cpa(ship1, ship2, past_timesteps, forecast_steps, prediction_method, interpolation_method, r_dcpa):
    predict_position1, predict_course1, predict_speed1, predict_time1 = dynamic_prediction(ship1, past_timesteps, forecast_steps, prediction_method)
    predict_position2, predict_course2, predict_speed2, predict_time2 = dynamic_prediction(ship2, past_timesteps, forecast_steps, prediction_method)

    x1_positions = [entry['x'] for entry in ship1] + [pos[0] for pos in predict_position1]  # lon
    y1_positions = [entry['y'] for entry in ship1] + [pos[1] for pos in predict_position1]  # lat
    x2_positions = [entry['x'] for entry in ship2] + [pos[0] for pos in predict_position2]
    y2_positions = [entry['y'] for entry in ship2] + [pos[1] for pos in predict_position2]

    if interpolation_method == 'cubic_hermite_spline':
        speed1 = [entry['SOG'] for entry in ship1] + list(predict_speed1)
        speed2 = [entry['SOG'] for entry in ship2] + list(predict_speed2)
        course1 = [entry['COG'] for entry in ship1] + list(predict_course1)
        course2 = [entry['COG'] for entry in ship2] + list(predict_course2)
        speed_x1 = [s * math.sin(math.radians(c)) for s, c in zip(speed1, course1)]
        speed_y1 = [s * math.cos(math.radians(c)) for s, c in zip(speed1, course1)]
        speed_x2 = [s * math.sin(math.radians(c)) for s, c in zip(speed2, course2)]
        speed_y2 = [s * math.cos(math.radians(c)) for s, c in zip(speed2, course2)]
        interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method, speed_x1, speed_y1)
        interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method, speed_x2, speed_y2)
    else:
        interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method)
        interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method)

    interpolated_xy1 = list(zip(interpolated_x1, interpolated_y1))
    interpolated_xy2 = list(zip(interpolated_x2, interpolated_y2))
    xy1_positions = list(zip(x1_positions, y1_positions))
    xy2_positions = list(zip(x2_positions, y2_positions))
    timestamps1 = [entry['timestamp'] for entry in ship1] + predict_time1
    timestamps2 = [entry['timestamp'] for entry in ship2] + predict_time2

    interpolated_timestamps1 = interpolate_timestamps(xy1_positions, interpolated_xy1, timestamps1)
    interpolated_timestamps2 = interpolate_timestamps(xy2_positions, interpolated_xy2, timestamps2)

    closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps1)
    interpolated_predict_xy1 = interpolated_xy1[closest_timestamp_index:]
    short_interpolated_xy1 = interpolated_xy1[:closest_timestamp_index]
    interpolated_predict_timestamps1 = interpolated_timestamps1[closest_timestamp_index:]
    closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps2)
    interpolated_predict_xy2 = interpolated_xy2[closest_timestamp_index:]
    short_interpolated_xy2 = interpolated_xy2[:closest_timestamp_index]
    interpolated_predict_timestamps2 = interpolated_timestamps2[closest_timestamp_index:]

    distances = []
    for i in range(len(interpolated_predict_xy1)):
        closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[i], interpolated_predict_timestamps2)
        distance = np.linalg.norm(np.asarray(interpolated_predict_xy1[i]) - np.asarray(interpolated_predict_xy2[closest_timestamp_index]))
        distances.append(distance)

    # print(f'Length of interpolated_predict_timestamps1: {len(interpolated_timestamps1)}')
    # print(f'Length of interpolated_predict_timestamps2: {len(interpolated_timestamps2)}')

    # if len(interpolated_timestamps1) != len(interpolated_timestamps2):
    #     raise ValueError("Interpolated timestamps lengths do not match")

    min_distance_index = distances.index(min(distances))
    cpa_position1 = interpolated_predict_xy1[min_distance_index]
    closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[min_distance_index], interpolated_predict_timestamps2)
    cpa_position2 = interpolated_predict_xy2[closest_timestamp_index]
    final_cpa = (cpa_position1, cpa_position2)
    cpa_distance = min(distances)
    time_to_cpa = interpolated_predict_timestamps1[min_distance_index] - ship1[-1]['timestamp']

    if r_dcpa:
        before_cpa_position1 = None
        before_cpa_position2 = None
        after_cpa_position1 = None
        after_cpa_position2 = None
        if min_distance_index > 0 and closest_timestamp_index > 0:
            before_cpa_position1 = interpolated_predict_xy1[min_distance_index - 1]
            before_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index - 1]
        else:
            after_cpa_position1 = interpolated_predict_xy1[min_distance_index + 1]
            after_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index + 1]

        if after_cpa_position1 != None:
            heading1 = clockwise_angle(after_cpa_position1[0] - cpa_position1[0], after_cpa_position1[1] - cpa_position1[1])
            heading2 = clockwise_angle(after_cpa_position2[0] - cpa_position2[0], after_cpa_position2[1] - cpa_position2[1])
        else:
            heading1 = clockwise_angle(cpa_position1[0] - before_cpa_position1[0], cpa_position1[1] - before_cpa_position1[1])
            heading2 = clockwise_angle(cpa_position2[0] - before_cpa_position2[0], cpa_position2[1] - before_cpa_position2[1])
        real_cpa_distance = real_dcpa(final_cpa, ship1[-1], heading1, ship2[-1], heading2)
    else:
        real_cpa_distance = None

    return final_cpa, cpa_distance, time_to_cpa, real_cpa_distance, predict_position1, predict_position2, interpolated_predict_xy1, interpolated_predict_xy2, short_interpolated_xy1, short_interpolated_xy2, predict_speed1, predict_course1, predict_speed2, predict_course2


def calculate_static_cpa(ship1, ship2, r_dcpa, ozt, security_radius):
    final_cpa, cpa_distance, time_to_cpa = calculate_standard_cpa(ship1[-1], ship2[-1])
    if ozt:
        ozt_points = obstacle_zone_by_target(security_radius, (ship1[-1]['x'], ship1[-1]['y']), (ship2[-1]['x'], ship2[-1]['y']), 
                                             ship2[-1]['heading'], ship1[-1]['SOG'], ship2[-1]['SOG'])
    else:
        ozt_points = None

    if r_dcpa:
        heading1 = ship1[-1]['heading']
        heading2 = ship2[-1]['heading']
        real_cpa_distance = real_dcpa(final_cpa, ship1[-1], heading1, ship2[-1], heading2)
    else:
        real_cpa_distance = None

    return final_cpa, cpa_distance, time_to_cpa, real_cpa_distance, ozt_points


def calculate_bayesian_cpa(ship1: Dict, ship2: Dict, past_timesteps: int, forecast_steps: int, prediction_method, interpolation_method, r_dcpa):
    ## Collect the variables needed for predictors >>
    # TODO: Refactor !
    # Ship 1 >
    sogs_ship1 = [entry['SOG'] for entry in ship1][-past_timesteps:]
    cogs_ship1 = [entry['COG'] for entry in ship1][-past_timesteps:]
    xs_ship1 = [entry['x'] for entry in ship1][-past_timesteps:]
    ys_ship1 = [entry['y'] for entry in ship1][-past_timesteps:]
    poss_ship1 = list(zip(xs_ship1, ys_ship1))
    time_ship1 = [entry['timestamp'] for entry in ship1][-past_timesteps:]
    
    # Ship 2 >
    sogs_ship2 = [entry['SOG'] for entry in ship2][-past_timesteps:]
    cogs_ship2 = [entry['COG'] for entry in ship2][-past_timesteps:]
    xs_ship2 = [entry['x'] for entry in ship2][-past_timesteps:]
    ys_ship2 = [entry['y'] for entry in ship2][-past_timesteps:]
    poss_ship2 = list(zip(xs_ship2, ys_ship2))
    time_ship2 = [entry['timestamp'] for entry in ship2][-past_timesteps:]
    
    predict_speed1, predict_course1, predict_position1
    
    predict_position1, predict_course1, predict_speed1, predict_time1 = dynamic_prediction(ship1, past_timesteps, forecast_steps, prediction_method)
    predict_position2, predict_course2, predict_speed2, predict_time2 = dynamic_prediction(ship2, past_timesteps, forecast_steps, prediction_method)

    x1_positions = [entry['x'] for entry in ship1] + [pos[0] for pos in predict_position1]  # lon
    y1_positions = [entry['y'] for entry in ship1] + [pos[1] for pos in predict_position1]  # lat
    x2_positions = [entry['x'] for entry in ship2] + [pos[0] for pos in predict_position2]
    y2_positions = [entry['y'] for entry in ship2] + [pos[1] for pos in predict_position2]

    if interpolation_method == 'cubic_hermite_spline':
        speed1 = [entry['SOG'] for entry in ship1] + list(predict_speed1)
        speed2 = [entry['SOG'] for entry in ship2] + list(predict_speed2)
        course1 = [entry['COG'] for entry in ship1] + list(predict_course1)
        course2 = [entry['COG'] for entry in ship2] + list(predict_course2)
        speed_x1 = [s * math.sin(math.radians(c)) for s, c in zip(speed1, course1)]
        speed_y1 = [s * math.cos(math.radians(c)) for s, c in zip(speed1, course1)]
        speed_x2 = [s * math.sin(math.radians(c)) for s, c in zip(speed2, course2)]
        speed_y2 = [s * math.cos(math.radians(c)) for s, c in zip(speed2, course2)]
        interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method, speed_x1, speed_y1)
        interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method, speed_x2, speed_y2)
    else:
        interpolated_x1, interpolated_y1 = interpolate_path(x1_positions, y1_positions, interpolation_method)
        interpolated_x2, interpolated_y2 = interpolate_path(x2_positions, y2_positions, interpolation_method)

    interpolated_xy1 = list(zip(interpolated_x1, interpolated_y1))
    interpolated_xy2 = list(zip(interpolated_x2, interpolated_y2))
    xy1_positions = list(zip(x1_positions, y1_positions))
    xy2_positions = list(zip(x2_positions, y2_positions))
    timestamps1 = [entry['timestamp'] for entry in ship1] + predict_time1
    timestamps2 = [entry['timestamp'] for entry in ship2] + predict_time2

    interpolated_timestamps1 = interpolate_timestamps(xy1_positions, interpolated_xy1, timestamps1)
    interpolated_timestamps2 = interpolate_timestamps(xy2_positions, interpolated_xy2, timestamps2)

    closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps1)
    interpolated_predict_xy1 = interpolated_xy1[closest_timestamp_index:]
    short_interpolated_xy1 = interpolated_xy1[:closest_timestamp_index]
    interpolated_predict_timestamps1 = interpolated_timestamps1[closest_timestamp_index:]
    closest_timestamp_index = find_closest_index(ship1[-1]['timestamp'], interpolated_timestamps2)
    interpolated_predict_xy2 = interpolated_xy2[closest_timestamp_index:]
    short_interpolated_xy2 = interpolated_xy2[:closest_timestamp_index]
    interpolated_predict_timestamps2 = interpolated_timestamps2[closest_timestamp_index:]

    distances = []
    for i in range(len(interpolated_predict_xy1)):
        closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[i], interpolated_predict_timestamps2)
        distance = np.linalg.norm(np.asarray(interpolated_predict_xy1[i]) - np.asarray(interpolated_predict_xy2[closest_timestamp_index]))
        distances.append(distance)

    min_distance_index = distances.index(min(distances))
    cpa_position1 = interpolated_predict_xy1[min_distance_index]
    closest_timestamp_index = find_closest_index(interpolated_predict_timestamps1[min_distance_index], interpolated_predict_timestamps2)
    cpa_position2 = interpolated_predict_xy2[closest_timestamp_index]
    final_cpa = (cpa_position1, cpa_position2)
    cpa_distance = min(distances)
    time_to_cpa = interpolated_predict_timestamps1[min_distance_index] - ship1[-1]['timestamp']

    if r_dcpa:
        before_cpa_position1 = None
        before_cpa_position2 = None
        after_cpa_position1 = None
        after_cpa_position2 = None
        if min_distance_index > 0 and closest_timestamp_index > 0:
            before_cpa_position1 = interpolated_predict_xy1[min_distance_index - 1]
            before_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index - 1]
        else:
            after_cpa_position1 = interpolated_predict_xy1[min_distance_index + 1]
            after_cpa_position2 = interpolated_predict_xy2[closest_timestamp_index + 1]

        if after_cpa_position1 != None:
            heading1 = clockwise_angle(after_cpa_position1[0] - cpa_position1[0], after_cpa_position1[1] - cpa_position1[1])
            heading2 = clockwise_angle(after_cpa_position2[0] - cpa_position2[0], after_cpa_position2[1] - cpa_position2[1])
        else:
            heading1 = clockwise_angle(cpa_position1[0] - before_cpa_position1[0], cpa_position1[1] - before_cpa_position1[1])
            heading2 = clockwise_angle(cpa_position2[0] - before_cpa_position2[0], cpa_position2[1] - before_cpa_position2[1])
        real_cpa_distance = real_dcpa(final_cpa, ship1[-1], heading1, ship2[-1], heading2)
    else:
        real_cpa_distance = None

    return final_cpa, cpa_distance, time_to_cpa, real_cpa_distance, predict_position1, predict_position2, interpolated_predict_xy1, interpolated_predict_xy2, short_interpolated_xy1, short_interpolated_xy2, predict_speed1, predict_course1, predict_speed2, predict_course2

