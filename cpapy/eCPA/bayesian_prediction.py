# import numpy as np
# from typing import List, Tuple, Dict, Any, Callable, Union


# from geopy.distance import geodesic

# from pykalman import KalmanFilter



# def _find_em_best_n_iter(kf: KalmanFilter, 
#                         measurements: np.ndarray, 
#                         max_iter: int, 
#                         em_vars: List[str]) -> int:
#     """
#     Finds the best number of iterations for the EM algorithm using the parameter change criterion.
    
#     Args:
#         kf (KalmanFilter): Kalman Filter object.
#         measurements (np.ndarray):  Measurements for fitting the Kalman Filter.
#                                     Each column should represent a different variable.
#         max_iter (int): Maximum number of iterations for the EM algorithm.
#         em_vars (List[str]): List of variables to optimize via the EM algorithm.
    
#     Returns:
#         int: Best number of iterations for the EM algorithm.
#     """
    
#     raise NotImplementedError("This function is not implemented yet.")


#     ## Initialise variables >>
#     best_n_iter = None
#     best_parameter_change = float('inf')

#     ## Find the best number of iterations for the EM algorithm using the parameter change criterion >>
#     for n_iter in range(1, max_iter + 1):
#         _kf = kf.em(measurements, n_iter=n_iter, em_vars=em_vars)
#         smoothed_state_means, _ = _kf.smooth(measurements)
#         filtered_state_means, _ = _kf.filter(measurements)

#         parameter_change = np.sum(np.abs(smoothed_state_means - filtered_state_means))

#         if parameter_change < best_parameter_change:
#             best_n_iter = n_iter
#             best_parameter_change = parameter_change

#     return best_n_iter

# def _predict_kalman_1d(measurements: Union[List, np.ndarray], num_predictions: int, max_iter: int = 10) -> np.ndarray:
    
#     raise NotImplementedError("This function is not implemented yet.")

#     ## Variables >>
#     # EM algorithm is that the algorithm lacks regularization, meaning that parameter values may diverge to infinity in order to make the measurements more likely.
#     # Thus it is important to choose which parameters to optimize via the em_vars parameter of KalmanFilter.
    
#     em_vars=['transition_matrices', 'transition_covariance', 'observation_matrices', 'observation_covariance']
    
#     measurements = np.asarray(measurements)  # Convert measurements to numpy array
#     if measurements.ndim != 1:  # If the measurements are 1D, reshape them to 2D
#         raise IndexError("The measurements must be 1D.")
    
#     # Initialise the kalman filter
#     initial_state_mean = measurements[0]  # the initial state of the system. It represents the best guess of the state variables at the beginning of the process. 
#     transition_matrix = [[1]]
#     observation_matrix = [[1]]
    
#     kf = KalmanFilter(
#         initial_state_mean=initial_state_mean,
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix,
#         n_dim_obs=2)  # n_dim_obs: number of observations space    
    
#     ## Find the best number of iterations for the EM algorithm using the parameter change criterion >>
    
#     return None

# def _predict_kalman_2d(measurements: Union[List, np.ndarray], num_predictions: int, max_iter: int = 10) -> np.ndarray:
    
#     raise NotImplementedError("This function is not implemented yet.")

#     ## Variables >>
#     # EM algorithm is that the algorithm lacks regularization, meaning that parameter values may diverge to infinity in order to make the measurements more likely.
#     # Thus it is important to choose which parameters to optimize via the em_vars parameter of KalmanFilter.
    
#     em_vars=['transition_matrices', 'transition_covariance', 'observation_matrices', 'observation_covariance']
    
#     measurements = np.asarray(measurements)  # Convert measurements to numpy array
#     if measurements.ndim != 1:  # If the measurements are 1D, reshape them to 2D
#         raise IndexError("The measurements must be 1D.")
    
#     # Initialise the kalman filter
#     initial_state_mean = measurements[0]  # the initial state of the system. It represents the best guess of the state variables at the beginning of the process. 
#     transition_matrix = [[1]]
#     observation_matrix = [[1]]
    
#     kf = KalmanFilter(
#         initial_state_mean=initial_state_mean,
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix,
#         n_dim_obs=2)  # n_dim_obs: number of observations space    
    
#     ## Find the best number of iterations for the EM algorithm using the parameter change criterion >>
    
#     return None
    
    
# def predict_kalman(SOG: List[float],
#                    COG: List[float],
#                    POS: List[Tuple[float, float]], 
#                    time_stamps: List[float],
#                    num_predictions: int,
#                    max_iter: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Predicts future states using the Kalman Filter algorithm.

#     Args:
#         SOG (List[float]): List of Speed Over Ground measurements.
#         COG (List[float]): List of Course Over Ground measurements.

        
#         num_predictions (int): Number of future state predictions to make.
#         max_iter (int, optional): Maximum number of iterations for the EM algorithm. Defaults to 10.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the predicted Speed Over Ground, Course Over Ground, and positions.
#     """
    
#     ## Variables >>
#     "EM algorithm is that the algorithm lacks regularization, meaning that parameter values may diverge to infinity in order to make the measurements more likely."
#     "Thus it is important to choose which parameters to optimize via the em_vars parameter of KalmanFilter."
#     em_vars=['transition_matrices', 'transition_covariance', 'observation_matrices', 'observation_covariance']
#     measurements = np.column_stack([SOG, COG])  # Stack the SOG and COG measurements
#     positions = np.asarray(POS)  # Convert positions to numpy array
#     start_pos = positions[-1]  # Get the last point of the training positions as the starting position for prediction
    
#     # ToDo(1): Change the time stamps to be in DateTime
#     # ToDo(2): Here the time stamps are assumed to be equidistanced. For that an interpolation must be included in the predictor.
#     delta_t = time_stamps[-1] - time_stamps[-2]  # Calculate the time difference between the last two time stamps
    
#     ## Initialize Kalman Filter >>
#     initial_state_mean = [SOG[0], COG[0]]
#     transition_matrix = np.eye(len(initial_state_mean))
#     observation_matrix = [[1, 0], [0, 1]]
    
#     kf = KalmanFilter(
#         initial_state_mean=initial_state_mean,
#         transition_matrices=transition_matrix,
#         observation_matrices=observation_matrix,
#         n_dim_obs=2)  # n_dim_obs: number of observations space    
    
#     ## Find the best number of iterations for the EM algorithm using the parameter change criterion >>
#     best_n_iter = None
#     best_parameter_change = float('inf')

#     for n_iter in range(1, max_iter):
#         # Perform the filtering and smoothing
#         _kf = KalmanFilter(
#             initial_state_mean=initial_state_mean,
#             transition_matrices=transition_matrix,
#             observation_matrices=observation_matrix,
#             n_dim_obs=2)
#         # _kf = _kf.em(measurements, n_iter=n_iter)
#         _kf = _kf.em(measurements, n_iter=n_iter, em_vars=em_vars)
#         smoothed_state_means, smoothed_state_covariances = _kf.smooth(measurements)
#         filtered_state_means, filtered_state_covariances = _kf.filter(measurements)

#         # Calculate the parameter change
#         parameter_change = np.sum(np.abs(smoothed_state_means - filtered_state_means))

#         # Check if this is the best fit so far
#         if parameter_change < best_parameter_change:
#             best_n_iter = n_iter
#             best_parameter_change = parameter_change

#     if best_n_iter is not None:
#         print("Best n_iter:", best_n_iter)
#     else:
#         print("Best n_iter is None, using max_iter instead.")
#         best_n_iter = max_iter
    
    
#     ## Fit the Kalman Filter to the data with the best_n_iter >>
#     # kf = kf.em(measurements, n_iter=best_n_iter)
#     kf = kf.em(measurements, n_iter=5, em_vars=em_vars)  # n_iter: number of iterations or steps used in the expectation-maximization (EM) algorithm.
#     (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    
#     # Predict future states
#     # future_state_means, future_state_covariances = kf.filter_update(filtered_state_means[-1], 
#     #                                                                 filtered_state_covariances[-1])
#     future_state_means: list = []
#     future_state_covariances: list = [] 
#     _predicted_SOG = []
#     _predicted_COG = []
#     _predicted_positions = [tuple(start_pos)]
#     _predicted_time_stamps = [time_stamps[-1]]
    
#     for _ in range(num_predictions):
#         # future_state_means, future_state_covariances = kf.filter_update(future_state_means, 
#         #                                                                 future_state_covariances)
#         future_state_means, future_state_covariances = kf.filter_update(filtered_state_means, 
#                                                                         filtered_state_covariances)
        
#         print(future_state_means)
        
#         _predicted_SOG.append(future_state_means[0])
#         _predicted_COG.append(future_state_means[1])
        
#         # Calculate new position
#         distance = future_state_means[0] * 1.852 / 3600  # Convert knots to km
#         bearing = future_state_means[1]
#         new_pos = geodesic(kilometers=distance).destination(_predicted_positions[-1], bearing)
#         _predicted_positions.append((new_pos.latitude, new_pos.longitude))
    
#     # Convert the lists into numpy arrays
#     predicted_SOG = np.array(_predicted_SOG)
#     predicted_COG = np.array(_predicted_COG)
#     predicted_positions = np.array(_predicted_positions) # np.array(_predicted_positions[1::]) 
    
#     return predicted_SOG, predicted_COG, predicted_positions
