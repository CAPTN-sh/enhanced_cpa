import yaml
import os
import sys
import csv
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import time

# # Add the parent directory of eCPA to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from plotting import TrajectoryVisualizer
from cpa_calculation import adjust_for_wind_and_current, calculate_dynamic_cpa, calculate_static_cpa
from interpolate_path import interpolate_scenarios
from standard_cpa import calculate_standard_cpa


class ECPA:
    def __init__(self, config_path, base_plot_path, train_models=True):
        self.config_path = config_path
        # self.ais_path = ais_path
        self.base_plot_path = base_plot_path
        self.train_models = train_models
        self.interpolation_method = ['linear', 'quadratic', 'cubic', 'cubic_spline', 'cubic_hermite_spline', 'pchip', 'akima']
        os.makedirs(self.base_plot_path, exist_ok=True)

    def load_yaml(self, filename):
        with open(os.path.join(self.config_path, filename), "r") as file:
            return yaml.safe_load(file)

    def extract_parameters(self, config, interpolation_method='linear', u=100, csv_data=None, scenario=None):
        self.plotting = config['plotting']
        self.dynamic = config['dynamic']
        self.wind_enabled = config['wind_enabled']
        self.current_enabled = config['current_enabled']
        self.prediction_method = config['prediction_method']
        self.forecast_steps = config['forecast_steps']
        self.past_timesteps = config['past_timesteps']
        self.r_dcpa = config['r_dcpa']
        self.ozt = config['ozt']
        self.axis_measure = config['axis_measure']
        self.security_radius = config['security_radius']
        self.gaussian_methods = config['gaussian_methods']
        self.anomalies_detect = config['anomalies_detect']
        
        if csv_data and scenario:
            all_data = {}

            interpolate_data = interpolate_scenarios(csv_data, interpolation_method, u=u)
            all_data[interpolation_method] = interpolate_data
            # print (f'all_data: {all_data}')
            self.final_cpa_x, self.final_cpa_y = all_data[interpolation_method][scenario]['final_cpa']
            self.final_cpa = all_data[interpolation_method][scenario]['final_cpa']
            self.final_tcpa = all_data[interpolation_method][scenario]['final_tcpa']
            self.final_dcpa = all_data[interpolation_method][scenario]['final_dcpa']


            self.own_ship_data = all_data[interpolation_method][scenario]['own_ship']
            self.target_ship_data = all_data[interpolation_method][scenario]['target_ship']
            # Find the own_ship point closest to the final_cpa coordinate.
            closest_point = min(self.own_ship_data, key=lambda point: (point['x'] - self.final_cpa_x)**2 + (point['y'] - self.final_cpa_y)**2)
            closest_timestamp = closest_point['timestamp']

            # Select all data points with a timestamp less than closest_point.
            self.ship1 = [point for point in self.own_ship_data if point['timestamp'] <= closest_timestamp]
            self.ship2 = [point for point in self.target_ship_data if point['timestamp'] <= closest_timestamp]

            # Remove the last 10 points from each ship's data.
            if len(self.ship1) > 40:
                self.ship1 = self.ship1[:-10]
            else:
                self.ship1 = self.ship1[:-5]

            if len(self.ship2) > 40:
                self.ship2 = self.ship2[:-10]
            else:
                self.ship2 = self.ship2[:-5]

            self.wind = None
            self.current = None
        else:
            self.ship1 = config['ship1']
            self.ship2 = config['ship2']
            self.wind = config['wind']
            self.current = config['current']

    def load_scenario_data_from_csv(self, csv_file):
        data = {}
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                scenario, ship_type, *values = row
                if scenario not in data:
                    data[scenario] = {}
                data[scenario][ship_type] = [eval(v) for v in values if v]
        return data
    
    def load_config(self, method, interpolation_method='akima', u=100, csv_file=None, scenario=None):
        if method == "statistical":
            # Load and extract parameters from statistical methods configuration
            config = self.load_yaml("statistical_methods.yaml")
        elif method == "bayesian":
            # Load and extract parameters from Bayesian filter configuration
            config = self.load_yaml("bayesian_filter.yaml")
        else:
            raise ValueError("Unsupported method specified")
        csv_data = self.load_scenario_data_from_csv(csv_file) if csv_file else None
        self.extract_parameters(config, interpolation_method, u, csv_data, scenario)

    def update_config(self, updates):
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid configuration key")
        return key, value

    def initialize_models(self):
        # Placeholder for initialize of data-driven models
        pass

    def apply_statistical_methods(self, pred_method, interpolation_method):
        start_time = time.time()  # Record the start time
        self.adjust_ships_for_wind_and_current()
        results = self.calculate_cpa(pred_method, interpolation_method)
        results = self.format_results(results)
        mse_results = self.compute_mse(self.ship1, self.ship2, results['final_cpa'], results['time_to_cpa'], results['cpa_distance'], results)
        results.update(mse_results)
        end_time = time.time()  # Record the end time
        results['elapsed_time'] = end_time - start_time  # Calculate the elapsed time
        return results

    def apply_bayesian_filter(self, pred_method, interpolation_method):
        start_time = time.time()  # Record the start time
        self.adjust_ships_for_wind_and_current()
        results = self.calculate_cpa(pred_method, interpolation_method)
        results = self.format_results(results)
        mse_results = self.compute_mse(self.ship1, self.ship2, results['final_cpa'], results['time_to_cpa'], results['cpa_distance'], results)
        results.update(mse_results)
        end_time = time.time()  # Record the end time
        results['elapsed_time'] = end_time - start_time  # Calculate the elapsed time
        return results

    def adjust_ships_for_wind_and_current(self):
        self.ship1 = adjust_for_wind_and_current(self.ship1, self.wind, self.current, self.wind_enabled, self.current_enabled)
        self.ship2 = adjust_for_wind_and_current(self.ship2, self.wind, self.current, self.wind_enabled, self.current_enabled)

    def calculate_cpa(self, pred_method, interpolation_method):
        if self.dynamic:
            return calculate_dynamic_cpa(self.ship1, self.ship2, self.past_timesteps, self.forecast_steps, pred_method, interpolation_method, self.r_dcpa)
        else:
            return calculate_static_cpa(self.ship1, self.ship2, self.r_dcpa, self.ozt, self.security_radius)

    # Anomaly detection
    def detect_anomalies(self, X, model, confidence_limit=0.55):
        if isinstance(model, gaussian_kde):
            densities = model.evaluate(X.T)
            max_density = np.max(densities)
            anomalies = densities < (max_density * confidence_limit)
            anomalies_X = X[anomalies]  # Get the coordinates of anomalies
            non_anomalies_X = X[~anomalies]
            print(f'densities: {densities}')
            print(f'max_density: {max_density}')
        elif isinstance(model, GaussianMixture):
            probs = np.exp(model.score_samples(X))
            max_probs = np.max(probs)
            anomalies = probs < (max_probs * confidence_limit)
            anomalies_X = X[anomalies]  # Get the coordinates of anomalies
            non_anomalies_X = X[~anomalies]
            print(f'max_probs: {max_probs}')
        else:
            raise ValueError("Unsupported model type for anomaly detection")
        print(f'max_probs: {max_probs}')
        print(f'anomalies: {anomalies_X}')
        return anomalies_X, non_anomalies_X

    def calculate_gaussian_results(self, results, method, detect_anomalies=True):
        eCPA_points = np.array([result['final_cpa'][0] for result in results.values()])
        if detect_anomalies:
            if method == 'kde':
                model = gaussian_kde(eCPA_points.T)
            elif method == 'gmm':
                model = GaussianMixture(n_components=1)
                model.fit(eCPA_points)
            else:
                raise ValueError("Unsupported method specified")
            anomalies_eCPA, eCPA_points = self.detect_anomalies(eCPA_points, model)
        else:
            anomalies_eCPA = np.zeros(len(eCPA_points), dtype=bool)
        min_x, max_x = eCPA_points[:, 0].min(), eCPA_points[:, 0].max()
        min_y, max_y = eCPA_points[:, 1].min(), eCPA_points[:, 1].max()
        num_points = 3000

        x = np.linspace(min_x, max_x, num_points)
        y = np.linspace(min_y, max_y, num_points)
        X, Y = np.meshgrid(x, y)
        grid_points = np.vstack([X.ravel(), Y.ravel()])

        if method == 'kde':
            kde = gaussian_kde(eCPA_points.T)
            Z = kde(grid_points).reshape(X.shape)
        elif method == 'gmm':
            gmm = GaussianMixture(n_components=1)
            gmm.fit(eCPA_points)
            Z = np.exp(gmm.score_samples(grid_points.T)).reshape(X.shape)

        max_index_flat = np.argmax(Z)
        max_index_2d = np.unravel_index(max_index_flat, Z.shape)
        max_x = X[max_index_2d]
        max_y = Y[max_index_2d]

        return {
            'method': method,
            'max_x': max_x,
            'max_y': max_y,
            'Z': Z,
            'X': X,
            'Y': Y,
            'eCPA_points': eCPA_points,
            'anomalies_eCPA': anomalies_eCPA
        }

    def format_results(self, results):
        if self.dynamic:
            keys = ["final_cpa", "cpa_distance", "time_to_cpa", "real_cpa_distance", "predict_position1", "predict_position2", 
                    "interpolated_predict_xy1", "interpolated_predict_xy2", "short_interpolated_xy1", "short_interpolated_xy2", 
                    "predict_speed1", "predict_course1", "predict_speed2", "predict_course2"]
        else:
            keys = ["final_cpa", "cpa_distance", "time_to_cpa", "real_cpa_distance", "ozt_points"]
        return dict(zip(keys, results))

    def enhance_data_driven_models(self):
        # Placeholder for enhancing data-driven models
        pass

    def visualize_trajectory(self, results, gaussian_results, method, combine_methods, prm, bg_img_path, csv_file=None, scenario=None):
        visualizer = TrajectoryVisualizer(self.axis_measure, bg_img_path)
        
        if csv_file is not None and scenario is not None:
            if combine_methods:
                if gaussian_results is not None:
                    plot_path = os.path.join(self.base_plot_path, f'{scenario}_ecpa_{method}_combined_gaussian.png')
                else:
                    plot_path = os.path.join(self.base_plot_path, f'{scenario}_ecpa_{method}_combined.png')
            else:
                plot_path = os.path.join(self.base_plot_path, f'{scenario}_ecpa_{method}_{prm}.png')
        else:
            if combine_methods:
                plot_path = os.path.join(self.base_plot_path, f'ecpa_{method}_combined.png')
            else:
                plot_path = os.path.join(self.base_plot_path, f'ecpa_{method}_{prm}.png')
        
        print(f'save_path: {plot_path}')
        # visualizer.plot_multiple_scenarios(results, plot_path)
        visualizer.plot_results(self.ship1, self.ship2, self.final_cpa_x, self.final_cpa_y, results, gaussian_results, plot_path, combine_methods, csv_file=csv_file)
        if gaussian_results is not None:
            visualizer.plot_gaussian_result(self.ship1, self.ship2, gaussian_results, plot_path)

    def compute_mse(self, own_ship, target_ship, final_cpa, final_tcpa, final_dcpa, results):
        mse_results = {
            'mse_ecpa': 0,
            'mse_cpa': 0,
            'mse_etcpa': 0,
            'mse_tcpa': 0,
            'mse_edcpa': 0,
            'mse_dcpa': 0
        }
        last_cpa = None  # Variable to store the last cpa value

        for i in range(1, len(own_ship)):
            mse_results['mse_ecpa'] += (results['final_cpa'][0][0] - final_cpa[0])**2 + (results['final_cpa'][0][1] - final_cpa[1])**2
            mse_results['mse_etcpa'] += (results['time_to_cpa'] - final_tcpa + own_ship[i]['timestamp'])**2
            mse_results['mse_edcpa'] += (results['cpa_distance'] - final_dcpa)**2

            cpa, cpa_distance, time_to_cpa = calculate_standard_cpa(own_ship[i], target_ship[i])
            last_cpa = cpa  # Save the current cpa value
            mse_results['mse_cpa'] += (cpa[0][0] - final_cpa[0])**2 + (cpa[0][1] - final_cpa[1])**2
            mse_results['mse_tcpa'] += (time_to_cpa - final_tcpa + own_ship[i]['timestamp'])**2
            mse_results['mse_dcpa'] += (cpa_distance - final_dcpa)**2

        for key in mse_results.keys():
            mse_results[key] /= (len(own_ship) - 1)
        mse_results['standard_cpa'] = last_cpa

        return mse_results

    def run(self, method, bg_img_path, u=100, config_updates=None, combine_methods=False, use_gaussian=False, csv_file=None, scenario=None):
        results = {}
        if combine_methods:
            # Apply Gaussian processing if specified
            if use_gaussian:
                print(f'self.interpolation_method: {self.interpolation_method}')
                for interpolation_method in self.interpolation_method:
                    self.load_config(method, interpolation_method, u, csv_file, scenario)
                    # self.initialize_models()
                    if config_updates:
                        _, prm = self.update_config(config_updates)
                    #results = {}
                    prm = None
                    for pred_method in self.prediction_method:
                        if method == "statistical":
                            result = self.apply_statistical_methods(pred_method, 'akima')
                        elif method == "bayesian":
                            result = self.apply_bayesian_filter(pred_method, 'akima')
                        else:
                            continue
                        results[interpolation_method + '_' + pred_method] = result
                gaussian_results = self.calculate_gaussian_results(results, self.gaussian_methods, self.anomalies_detect) 
            else:
                self.load_config(method, 'akima', u, csv_file, scenario)
                # self.initialize_models()
                if config_updates:
                    _, prm = self.update_config(config_updates)
                #results = {}
                prm = None
                gaussian_results = None
                interpolation_method = self.interpolation_method[-1]
                for pred_method in self.prediction_method:  
                    if method == "statistical":
                        result = self.apply_statistical_methods(pred_method, interpolation_method)
                    elif method == "bayesian":
                        result = self.apply_bayesian_filter(pred_method, interpolation_method)
                    else:
                        continue
                    results[pred_method] = result

        else:
            self.load_config(method,  'akima', u, csv_file, scenario)
            gaussian_results = None
            # self.initialize_models()
            if config_updates:
                _, prm = self.update_config(config_updates)
            #results = {}
            if method == "statistical":
                if self.dynamic:
                    results = self.apply_statistical_methods(pred_method, interpolation_method)
                else:
                    results = self.apply_statistical_methods(None, None)
            elif method == "bayesian":
                if self.dynamic:
                    results = self.apply_bayesian_filter(pred_method, interpolation_method)
                else:
                    results = self.apply_bayesian_filter(None, None)

        
        if results:
            self.print_results(results, method, combine_methods)
            self.visualize_trajectory(results, gaussian_results, method, combine_methods, prm, bg_img_path, csv_file, scenario)

        return results, self.ship1, self.ship2, self.final_cpa, self.own_ship_data, self.target_ship_data, gaussian_results # gaussian_results can be removed
        
    def print_results(self, results, method, combine_methods):
        print(f"Prediction approach: {method}")

        if combine_methods:
            for pred_method, result in results.items():
                print(f"\nPrediction method: {pred_method}")
                print(f"Closest Point of Approach (CPA): {result['final_cpa']}")
                print(f"Distance at CPA (DCPA): {result['cpa_distance']}")
                if self.r_dcpa:
                    print(f"Real Distance at CPA (rDCPA): {result['real_cpa_distance']}")
                print(f"Time to CPA (TCPA): {result['time_to_cpa']}")
                if not self.dynamic and self.ozt:
                    print(f"Obstacle Zone to Target (OZT): {result.get('ozt_points')}")
                print(f"Elapsed time: {result['elapsed_time']} seconds")
                print(f"MSE eCPA: {result['mse_ecpa']}")
                print(f"MSE CPA: {result['mse_cpa']}")
                print(f"MSE eTCPA: {result['mse_etcpa']}")
                print(f"MSE TCPA: {result['mse_tcpa']}")
                print(f"MSE eDCPA: {result['mse_edcpa']}")
                print(f"MSE DCPA: {result['mse_dcpa']}")
                print(f"Standard CPA: {result['standard_cpa']}")
        else:
            print(f"Closest Point of Approach (CPA): {results['final_cpa']}")
            print(f"Distance at CPA (DCPA): {results['cpa_distance']}")
            if self.r_dcpa:
                print(f"Real Distance at CPA (rDCPA): {results['real_cpa_distance']}")
            print(f"Time to CPA (TCPA): {results['time_to_cpa']}")
            if not self.dynamic and self.ozt:
                print(f"Obstacle Zone to Target (OZT): {results.get('ozt_points')}")
            print(f"Elapsed time: {results['elapsed_time']} seconds")
            print(f"MSE eCPA: {results['mse_ecpa']}")
            print(f"MSE CPA: {results['mse_cpa']}")
            print(f"MSE eTCPA: {results['mse_etcpa']}")
            print(f"MSE TCPA: {results['mse_tcpa']}")
            print(f"MSE eDCPA: {results['mse_edcpa']}")
            print(f"MSE DCPA: {results['mse_dcpa']}")
            print(f"Standard CPA: {result['standard_cpa']}")