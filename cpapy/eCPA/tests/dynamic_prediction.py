"""

Compute a prediction of the given ships dynamic positioning (non-linear).

Author: Tom Beyer

"""    

import math
import numpy as np
import statsmodels.api as sm
import warnings

from scipy import integrate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Make a prediction by the Triple Exponential Smoothing method like described by Sang et.al. 2016
def predict_by_triple_exp_smoothing(data, predictions=1, alpha=0.65):
    # Check if there is enough data
    if len(data) < 2:
        raise ValueError("Input array must have at least two values.")
    # Initialize the single, double and triple smoothing value to the first value
    s1 = s2 = s3 = data[0]
    # Compute the predictions by running through the whole time series - and beyond
    for i in range(len(data) - 1):
        s1 = alpha * data[i + 1] + (1 - alpha) * s1
        s2 = alpha * s1 + (1 - alpha) * s2
        s3 = alpha * s2 + (1 - alpha) * s3
    # Make the desired predictions
    predicts = []
    a = 3 * s1 - 3 * s2 + s3
    b = alpha ** 2 / 2 * (1 - alpha) ** 2 * ((6 - 5 * alpha) * s1 - (10 - 8 * alpha) * s2 + (4 - 3 * alpha) * s3)
    c = alpha ** 2 / (1 - alpha) ** 2 * (s1 - 2 * s2 + s3)
    for i in range(predictions):
        predict = a + (i + 1) * b + 0.5 * (i + 1) ** 2 * c
        predicts.append(predict)

    return predicts


# Make a predicition for the future course
def dynamic_prediction(ship, past_timesteps, forecast_steps, prediction_method):
    # Gather data for the relevant number of last timesteps
    speed = [entry['SOG'] for entry in ship][-past_timesteps:]
    course = [entry['COG'] for entry in ship][-past_timesteps:]
    time = [entry['timestamp'] for entry in ship][-past_timesteps:]
    # Initizalize result
    predict_speed = []
    predict_course = []
    predict_position = []
    # Use Double Exponential Smoothing
    data = [speed, course]
    forecast_result = []
    if prediction_method == 'double_exponential':
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for i in range(len(data)):
                model = sm.tsa.ExponentialSmoothing(data[i], trend="add", seasonal=None)
                result = model.fit()
                forecast = result.forecast(steps=forecast_steps)
                forecast_result.append(forecast)
        # Save resulting predictions
        predict_speed = forecast_result[0]
        predict_course = forecast_result[1]
    # Use Triple Exponential Smoothing
    elif prediction_method == 'triple_exponential':
        for i in range(len(data)):
            forecast = predict_by_triple_exp_smoothing(data[i], forecast_steps, alpha=0.65)
            forecast_result.append(forecast)
        # Save resulting predictions
        predict_speed = forecast_result[0]
        predict_course = forecast_result[1]
    # Otherwise find a Polynomial to describe the data
    elif prediction_method == 'polynomial':
        # Create a PolynomialFeatures instance to transform the time steps
        polynomial_features = PolynomialFeatures(degree=2)
        # Define the necessary time steps
        time_steps = (np.arange(past_timesteps)).reshape(-1, 1)  # ordinal time steps
        # Transform the time steps to include polynomial features
        time_steps_poly = polynomial_features.fit_transform(time_steps)
        # Create a LinearRegression model
        model = LinearRegression()
        for i in range(len(data)):
            target = np.asarray(data[i])  # dependent variable (target)
            # Fit the model to the transformed time steps
            model.fit(time_steps_poly, target)
            # Generate predictions using the model
            x_prediction = np.arange(0, past_timesteps + forecast_steps, 1).reshape(-1, 1)
            x_prediction_polynomial = polynomial_features.transform(x_prediction)
            y_prediction = model.predict(x_prediction_polynomial)
            forecast_result.append(y_prediction)
        # Save resulting predictions
        predict_speed = forecast_result[0][past_timesteps:]
        predict_course = forecast_result[1][past_timesteps:]
    # Compute the distances in the forecast
    predict_position = [(ship[-1]['x'], ship[-1]['y'])]  # initialize with last old position
    for i in range(forecast_steps):
        # Actualize the data for the next position forecasting step in the loop
        speed.append(predict_speed[i])
        course.append(predict_course[i])
        time.append(time[-1] + (time[-1] - time[-2]))
        # Horizontal distance ship 1
        def dist_x(t): 
            return (speed[-2] + (speed[-1] - speed[-2]) * (t - time[-2]) / (time[-1] - time[-2])) * \
                    math.sin(math.radians(course[-2] + (course[-1] - course[-2]) * (t - time[-2]) / (time[-1] - time[-2])))
        result_dist_x1, error_dist_x1 = integrate.quad(dist_x, time[-2], time[-1])
        # Vertical distance ship 1
        def dist_y(t): 
            return (speed[-2] + (speed[-1] - speed[-2]) * (t - time[-2]) / (time[-1] - time[-2])) * \
                    math.cos(math.radians(course[-2] + (course[-1] - course[-2]) * (t - time[-2]) / (time[-1] - time[-2])))
        result_dist_y1, error_dist_y1 = integrate.quad(dist_y, time[-2], time[-1])
        # Save resulting predicted position
        predict_position.append((predict_position[-1][0] + result_dist_x1, predict_position[-1][1] + result_dist_y1))
    # Save time data
    predict_time = time[-forecast_steps:]
    # Reset the data to the state previous to the position forecasting loop
    speed = speed[:-forecast_steps]
    course = course[:-forecast_steps]
    time = time[:-forecast_steps]
    # Clear predicted position data (first entry is already part of the overall position data)
    predict_position = predict_position[1:]

    return predict_position, predict_course, predict_speed, predict_time