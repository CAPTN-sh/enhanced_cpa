"""

Compute the standard (linear) Closest Point of Approach (CPA).

Author: Tom Beyer

"""

import numpy as np
import math

# Calculate the normal CPA (Closest Point of Approach seen from ship 1).
def calculate_standard_cpa(ship1, ship2):
    # Extract relevant information for each ship
    x1, y1, speed1, course1 = ship1['x'], ship1['y'], ship1['SOG'], ship1['heading']
    x2, y2, speed2, course2 = ship2['x'], ship2['y'], ship2['SOG'], ship2['heading']

    # Calculate relative velocity components.
    # If rel_SOG_x and rel_SOG_y are both positive, the vessels are moving closer to each other.
    # If they are both negative, they are moving away from each other.
    # If one is positive and one negative, one is moving towards the other while the other is moving away.
    rel_SOG_x = speed1 * math.sin(math.radians(course1)) - speed2 * math.sin(math.radians(course2))  # the horizontal component
    rel_SOG_y = speed1 * math.cos(math.radians(course1)) - speed2 * math.cos(math.radians(course2))  # the vertical component

    # Calculate relative position
    rel_x = x2 - x1
    rel_y = y2 - y1

    # Calculate time to CPA
    if rel_SOG_x == 0.0 and rel_SOG_y == 0.0:
        time_to_cpa = float('inf')
    else:
        time_to_cpa = (rel_x * rel_SOG_x + rel_y * rel_SOG_y) / (rel_SOG_x ** 2 + rel_SOG_y ** 2)

    # Calculate CPA coordinates
    cpa_x1 = x1 + speed1 * math.sin(math.radians(course1)) * time_to_cpa
    cpa_y1 = y1 + speed1 * math.cos(math.radians(course1)) * time_to_cpa
    cpa_x2 = x2 + speed2 * math.sin(math.radians(course2)) * time_to_cpa
    cpa_y2 = y2 + speed2 * math.cos(math.radians(course2)) * time_to_cpa
    cpa = [(cpa_x1, cpa_y1), (cpa_x2, cpa_y2)]

    # Calculate the distance at CPA
    # cpa_distance = math.sqrt((cpa_x - x1) ** 2 + (cpa_y - y1) ** 2)
    cpa_distance = np.linalg.norm(np.asarray(cpa[0]) - np.asarray(cpa[1]))

    return cpa, cpa_distance, time_to_cpa
