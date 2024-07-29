"""

Compute Obstacle Zone by Target (OZT) after Imazu (2017).

Author: Tom Beyer

"""

import math
import numpy as np

# Computes the x- and y-components of a vessels movement.
def compute_components(speed, course):
    # Convert angle to radians
    angle_rad = math.radians(course)
    # Compute x- and y-components
    x_component = speed * math.sin(angle_rad)
    y_component = speed * math.cos(angle_rad)

    return x_component, y_component


# Calculates the Euclidean distance between two two-dimensional points.
def distance(point1, point2):

    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# Finds the two two-dimensional points that a furthest away from a given point.
def find_furthest_points(points, given_point):
    # Determine and sort the distances between all points and the given one
    distances = [(distance(given_point, point), point) for point in points]
    distances.sort(key=lambda x: x[0], reverse=True)
    # Filter the two furthest points
    furthest_points = [distances[0][1], distances[1][1]]
    
    return furthest_points


# Checks if a given point is on a circle, defined by its center and radius.
def is_point_on_circle(x, y, center_x, center_y, radius):
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    return distance_squared == radius ** 2


# Finds the two tangent lines along a circle that go through a given external point.
def find_tangent_points(center_x, center_y, radius, external_x, external_y):
    distance = math.sqrt((external_x - center_x)**2 + (external_y - center_y)**2)
    # Check if there are tagents at all
    if distance <= radius:
        # print("The point is inside or on the circle. No tangent lines.")

        return None, None
    
    # Calculate the angle (theta) between the center of the circle and the external point
    theta = math.atan2(external_y - center_y, external_x - center_x)
    # Compute the tangent points
    tangent_point1 = (center_x + radius * math.cos(theta + math.pi / 2), center_y + radius * math.sin(theta + math.pi / 2))
    tangent_point2 = (center_x + radius * math.cos(theta - math.pi / 2), center_y + radius * math.sin(theta - math.pi / 2))

    return tangent_point1, tangent_point2


# Finds the linear equation of two given two-dimensional points.
def linear_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Calculate the slope
    slope = (y2 - y1) / (x2 - x1)
    # Calculate the y-intercept
    intercept = y1 - slope * x1

    return slope, intercept


# Finds the intersection of two lines given by their linear equations.
def find_intersection(slope1, intercept1, slope2, intercept2):
    # Solve for x_intersect
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    # Substitute x_intersect into either equation to find y_intersect
    y_intersect = slope1 * x_intersect + intercept1

    return x_intersect, y_intersect


# Shifts a line (defined by its linear equation) so that it goes through a given point.
def shift_line_through_point(slope, given_point):
    # Calculate the new intercept with the y-axis
    new_intercept = slope * (-given_point[0]) + given_point[1]

    return slope, new_intercept


# Finds the intersection point of a line with a circle.
def find_intersection_points(line_slope, line_intercept, circle_center, circle_radius):
    # Coefficients of the quadratic equation
    a = 1 + line_slope ** 2
    b = -2 * (circle_center[0] - line_slope * (line_intercept - circle_center[1]))
    c = circle_center[0] ** 2 + (line_intercept - circle_center[1]) ** 2 - circle_radius ** 2
    # Calculate the discriminant
    delta = b ** 2 - 4 * a * c
    # Check for real solutions
    if delta >= 0:
        # Calculate the x-coordinates of the intersection points
        x1 = (-b + np.sqrt(delta)) / (2 * a)
        x2 = (-b - np.sqrt(delta)) / (2 * a)
        # Calculate the corresponding y-coordinates
        y1 = line_slope * x1 + line_intercept
        y2 = line_slope * x2 + line_intercept
        # Save the results
        intersection_points = [(x1, y1), (x2, y2)]

        return intersection_points
    
    # No real solutions, circles and lines do not intersect
    else:

        return []

    
# Computes the Obstacle Zone by Target (OZT).
def obstacle_zone_by_target(security_radius, position1, position2, course2, speed1, speed2):
    # Find the tangent points around the security zone circle around the ASV projected from the other ships position
    tangent1, tangent2 = find_tangent_points(position1[0], position1[1], security_radius, position2[0], position2[1])
    if tangent1 is None or tangent2 is None:

        return None
    
    # Calculate the components of the other vessels movement
    speed2_x, speed2_y = compute_components(speed2, course2)
    # 'Draw' the motion vector of the other ship behind it
    position2_minus_motion = (position2[0] - speed2_x, position2[1] - speed2_y)
    # Represent the already computed tangents via slope and interception with the y-axis (so, as linear equation)
    slope1, intercept1 = linear_equation(tangent1, position2)
    slope2, intercept2 = linear_equation(tangent2, position2)
    # Find the intersections of the tangents with the a new circle (radius = speed of ASV) centered at the other vessels
    # position minus its movement vector
    intersection_points = []
    intersection_points += find_intersection_points(slope1, intercept1, position2_minus_motion, speed1)
    intersection_points += find_intersection_points(slope2, intercept2, position2_minus_motion, speed1)
    # Check if intersection points were found
    if len(intersection_points) == 0:

        return None
    
    # Find the furthest two intersections; the ones that are opposite to the ASV, seen from the other vessels perspective
    furthest_intersections = find_furthest_points(intersection_points, position1)
    # Compute the lines that go from the other vessels position minus its movement vector and the above found furthest intersections
    collision_slope1, collision_intercept1 = linear_equation(position2_minus_motion, furthest_intersections[0])
    collision_slope2, collision_intercept2 = linear_equation(position2_minus_motion, furthest_intersections[1])
    # Shift the lines to the position of the ASV (from the other vessels position minus its movement vector)
    result_collision_slope1, result_collision_intercept1 = shift_line_through_point(collision_slope1, position1)
    result_collision_slope2, result_collision_intercept2 = shift_line_through_point(collision_slope2, position1)
    # Represent the course of the other vessel as linear equation
    course2_slope, course2_intercept = linear_equation(position2, (position2[0] + speed2_x, position2[1] + speed2_y))
    # Determine the line (defined by two points on it) between the intersections of the other ships course 
    # with those shifted collision intersection lines
    course_collision_intersection1 = find_intersection(course2_slope, course2_intercept, result_collision_slope1, result_collision_intercept1)
    course_collision_intersection2 = find_intersection(course2_slope, course2_intercept, result_collision_slope2, result_collision_intercept2)

    return [course_collision_intersection1, course_collision_intersection2]
