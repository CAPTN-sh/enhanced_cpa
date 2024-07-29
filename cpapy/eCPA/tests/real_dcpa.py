"""

Compute the real Distance to the Closest Point of Approach (real DCPA).

Author: Tom Beyer

"""

import numpy as np
import math

# Converts an angle from the 360-degree realm to the equivalent angle in the 180-degree realm.
def convert_to_180_degrees(angle_360_degrees):
    # Ensure the angle is in the range [0, 360)
    angle_360_degrees = angle_360_degrees % 360
    # Convert to the range [-180, 180)
    if angle_360_degrees >= 180:
        angle_180_degrees = angle_360_degrees - 360
    else:
        angle_180_degrees = angle_360_degrees

    return angle_180_degrees


# Computes the overlap distance used for the real DCPA.
def overlap_distance(cpa, heading, bow, starboard, stern, portside):
    # Initialize result variable
    result = None
    # Calculate the angle (gamma) between the line spannend by the CPAs and the heading
    result_angle = (math.degrees(math.atan2(cpa[1][0] - cpa[0][0], cpa[1][1] - cpa[0][1])) + 360) % 360
    gamma = convert_to_180_degrees(result_angle - heading)

    # Find the right sector of the ship and the corresponding formula
    # Bow region
    if gamma >= math.degrees(-math.atan(portside / bow)) and gamma <= math.degrees(math.atan(starboard / bow)):
        result = bow / math.cos(math.radians(gamma))
    # Starboard region
    elif gamma > math.degrees(math.atan(starboard / bow)) and gamma < math.degrees(math.pi - math.atan(starboard / stern)):
        result = starboard / math.cos(0.5 * math.pi - math.radians(gamma))
    # Stern region
    elif gamma >= math.degrees(math.pi - math.atan(starboard / stern)) and gamma < math.degrees(math.pi) or \
         gamma >= math.degrees(-math.pi) and gamma <= math.degrees(-math.pi + math.atan(portside / stern)):
        result = stern / math.cos(math.pi - math.radians(gamma))
    # Portside region
    elif gamma > math.degrees(-math.pi + math.atan(portside / stern)) and gamma < math.degrees(-math.atan(portside / bow)):
        result = portside / math.cos(1.5 * math.pi - math.radians(gamma))

    return result


# Computes the real DCPA.
def real_dcpa(cpa, ship1, heading1, ship2, heading2):
    # Determine the distance between the CPAs of both vessels
    dist = np.linalg.norm(np.asarray(cpa[0]) - np.asarray(cpa[1]))
    # Compute the overlap distances of both vessels
    m1 = overlap_distance(cpa, heading1, ship1['bow'], ship1['starboard'], ship1['stern'], ship1['portside'])
    m2 = overlap_distance((cpa[1], cpa[0]), heading2, ship2['bow'], ship2['starboard'], ship2['stern'], ship2['portside'])
    # Calculate the real DCPA
    real_dcpa = dist - m1 - m2

    return real_dcpa