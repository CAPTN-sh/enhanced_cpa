import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import random
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

import scienceplots  # https://github.com/garrettj403/SciencePlots?tab=readme-ov-file
plt.style.use(['science', 'notebook', 'ieee'])  # , 'ieee','grid'

class TrajectoryVisualizer:
    def __init__(self, axis_measure, img_path):
        self.axis_measure = axis_measure
        self.img_path = img_path

    def plot_ships_and_paths(self, ax, ship_positions, color, label, legend_handles, add_legend):
        x = [ship['x'] for ship in ship_positions]
        y = [ship['y'] for ship in ship_positions]
        ax.plot(x, y, color=color, zorder=2, linestyle='-', linewidth=1.0, label=label if add_legend else "")

        if add_legend:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

        # for ship in ship_positions:
        #     ax.arrow(ship['x'], ship['y'], ship['SOG'] * math.sin(math.radians(ship['heading'])),
        #              ship['SOG'] * math.cos(math.radians(ship['heading'])),
        #              head_width=5, head_length=5, color=color, zorder=2)
        # Only draw arrow for the first point
        first_ship = ship_positions[0]
        ax.arrow(first_ship['x'], first_ship['y'], first_ship['SOG'] * math.sin(math.radians(first_ship['heading'])),
                 first_ship['SOG'] * math.cos(math.radians(first_ship['heading'])),
                 head_width=5, head_length=5, color=color, zorder=2)
        
    def plot_predicted_paths(self, ax, prediction_positions, predict_speed, predict_course, base_color, label, legend_handles, add_legend):
        color = self.get_random_color_within_shade(base_color)
        # color = self.get_fixed_color_within_shade(base_color)
        x = [predict[0] for predict in prediction_positions]
        y = [predict[1] for predict in prediction_positions]
        ax.plot(x, y, color=color, alpha=0.5, linestyle='--', linewidth=1.0, label=label if add_legend else "")

        if add_legend:
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, linestyle='--', label=label))

        # for i, predict in enumerate(prediction_positions):
        #     ax.arrow(predict[0], predict[1], predict_speed[i] * math.sin(math.radians(predict_course[i])),
        #              predict_speed[i] * math.cos(math.radians(predict_course[i])),
        #              head_width=5, head_length=5, color=color, linestyle='--', zorder=2)
        # Only draw arrow for the first point
        first_predict = prediction_positions[0]
        ax.arrow(first_predict[0], first_predict[1], predict_speed[0] * math.sin(math.radians(predict_course[0])),
                 predict_speed[0] * math.cos(math.radians(predict_course[0])),
                 head_width=5, head_length=5, color=color, linestyle='--', zorder=2)

    def get_random_color_within_shade(self, base_color):
        base_rgba = mcolors.to_rgba(base_color)
        h, s, v = mcolors.rgb_to_hsv(base_rgba[:3])
        
        # Adjust hue, saturation and vibrance to produce more distinguishable colors
        new_h = (h + random.uniform(-0.1, 0.1)) % 1.0  
        new_s = min(max(s + random.uniform(-0.3, 0.3), 0), 1)  
        new_v = min(max(v + random.uniform(-0.3, 0.3), 0), 1)  
        
        new_rgb = mcolors.hsv_to_rgb((new_h, new_s, new_v))
        new_rgba = (*new_rgb, base_rgba[3])
        return new_rgba
    
    def get_fixed_color_within_shade(base_color, hue_adjust=0.05, saturation_adjust=0.2, vibrance_adjust=0.2):
        while True:
            base_rgba = mcolors.to_rgba(base_color)
            h, s, v = mcolors.rgb_to_hsv(base_rgba[:3])
            
            # Adjust hue, saturation and vibrance by fixed amounts
            new_h = (h + hue_adjust) % 1.0  
            new_s = min(max(s + saturation_adjust, 0), 1)  
            new_v = min(max(v + vibrance_adjust, 0), 1)  
            
            new_rgb = mcolors.hsv_to_rgb((new_h, new_s, new_v))
            new_rgba = (*new_rgb, base_rgba[3])
            
            # Validate the new_rgba format
            if len(new_rgba) == 4 and all(0 <= value <= 1 for value in new_rgba):
                break
            
            # Optionally adjust the adjustments slightly to avoid infinite loop
            hue_adjust = random.uniform(-0.1, 0.1)
            saturation_adjust = random.uniform(-0.3, 0.3)
            vibrance_adjust = random.uniform(-0.3, 0.3)
    
        return new_rgba

    def plot_interpolated_paths(self, ax, result, ship1_color, ship2_color, label, legend_handles, add_legend):
        if 'interpolated_predict_xy1' in result and result['interpolated_predict_xy1'] and 'interpolated_predict_xy2' in result and result['interpolated_predict_xy2']:
            interpolated_predict_xy1 = result['interpolated_predict_xy1']
            interpolated_predict_xy2 = result['interpolated_predict_xy2']
            ax.plot([ele[0] for ele in interpolated_predict_xy1], [ele[1] for ele in interpolated_predict_xy1], color=ship1_color, linestyle='--', linewidth=0.8, alpha=0.5)
            ax.plot([ele[0] for ele in interpolated_predict_xy2], [ele[1] for ele in interpolated_predict_xy2], color=ship2_color, linestyle='--', linewidth=0.8, alpha=0.5)

        if 'short_interpolated_xy1' in result and result['short_interpolated_xy1'] and 'short_interpolated_xy2' in result and result['short_interpolated_xy2']:
            short_interpolated_xy1 = result['short_interpolated_xy1']
            short_interpolated_xy2 = result['short_interpolated_xy2']
            ax.plot([ele[0] for ele in short_interpolated_xy1], [ele[1] for ele in short_interpolated_xy1], color=ship1_color, linewidth=0.8, alpha=0.7)
            ax.plot([ele[0] for ele in short_interpolated_xy2], [ele[1] for ele in short_interpolated_xy2], color=ship2_color, linewidth=0.8, alpha=0.7)

        if 'final_cpa' in result and result['final_cpa']:
            final_cpa = result['final_cpa']
            random_color = self.get_random_color_within_shade('gold')
            ax.plot(final_cpa[0][0], final_cpa[0][1], "*", markersize=4, markeredgewidth=2, color='white')
            ax.plot(final_cpa[0][0], final_cpa[0][1], "*", markersize=3, markeredgewidth=1.5, color=random_color)
            if add_legend:
                legend_handles.append(plt.Line2D([0], [0], marker='*', color=random_color, lw=0, markersize=10, label=f'{label} CPA'))

    def plot_wind_current(self, ax, data, data_type, color1, color2):
        if data_type == "wind":
            position = 275
            text = "Wind"
        else:
            position = 225
            text = "Current"

        speed = data[-1]['speed'] * (3.6 / 1.852 if data_type == "wind" else 0.036 / 1.852)
        heading = (data[-1]['direction'] + 180) % 360
        ax.arrow(position, 25, speed * math.sin(math.radians(heading)),
                 speed * math.cos(math.radians(heading)), 
                 head_width=6, head_length=6, color='k', linewidth=3)
        ax.arrow(position, 25, speed * math.sin(math.radians(heading)),
                 speed * math.cos(math.radians(heading)), 
                 head_width=5, head_length=5, color=color2, linewidth=2)
        ax.plot([position - 25, position - 25], [0, 50], color='k', linewidth=0.8)
        ax.plot([position - 25, position + 25], [50, 50], color='k', linewidth=0.8)
        ax.text(position - 11, 52, text, fontsize=8, color='black')

    def plot_results(self, ship1, ship2,final_cpa_x, final_cpa_y, results, gaussian_results, save_path, combine_methods, wind=None, current=None, wind_enabled=False, current_enabled=False, csv_file=None):
        fig, ax = plt.subplots()
        ax.axis(self.axis_measure)
        ax.grid(True)
        plt.xticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
        plt.yticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
        plt.xlabel('Nautical Miles [NM]')
        plt.ylabel('Nautical Miles [NM]')
        if csv_file is None:
            print(csv_file)
            img = plt.imread(self.img_path)
            ax.imshow(img, extent=self.axis_measure)

        if wind_enabled and wind:
            self.plot_wind_current(ax, wind, "wind", 'k', 'b')

        if current_enabled and current:
            self.plot_wind_current(ax, current, "current", 'k', 'w')

        ship1_base_color = 'blue'
        ship2_base_color = 'green'

        legend_handles = []

        # Add a legend entry for the current position of the boat
        self.plot_ships_and_paths(ax, ship1, ship1_base_color, "Own", legend_handles, add_legend=True)
        self.plot_ships_and_paths(ax, ship2, ship2_base_color, "Target", legend_handles, add_legend=True)
        add_legend = True

        if combine_methods:
            for _, (pred_method, result) in enumerate(results.items()):
                self.plot_predicted_paths(ax, result['predict_position1'], result['predict_speed1'], result['predict_course1'], ship1_base_color, f"{pred_method} (Own)", legend_handles, add_legend)
                self.plot_predicted_paths(ax, result['predict_position2'], result['predict_speed2'], result['predict_course2'], ship2_base_color, f"{pred_method} (Target)", legend_handles, add_legend)
                self.plot_interpolated_paths(ax, result, ship1_base_color, ship2_base_color, f"{pred_method}", legend_handles, add_legend)

            # if gaussian_results is not None:
            #     self.plot_gaussian_model(ax, gaussian_results, legend_handles, add_legend)

        else:
            self.plot_predicted_paths(ax, results['predict_position1'], results['predict_speed1'], results['predict_course1'], ship1_base_color, "predicted (Own)", legend_handles, add_legend)
            self.plot_predicted_paths(ax, results['predict_position2'], results['predict_speed2'], results['predict_course2'], ship2_base_color, "predicted (Target)", legend_handles, add_legend)
            self.plot_interpolated_paths(ax, results, ship1_base_color, ship2_base_color, f"", legend_handles, add_legend)
        # ax.plot(final_cpa_x, final_cpa_y, "x", markersize=5.5, markeredgewidth=4, color='white', alpha=0.7)
        ax.plot(final_cpa_x, final_cpa_y, "x", markersize=5, markeredgewidth=2, color='red', alpha=0.5)
        if add_legend:
                legend_handles.append(plt.Line2D([0], [0], marker='x', color='red', lw=0, markersize=7, label=f'Real CPA'))

        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=True, framealpha=0.5, edgecolor='black')
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    def plot_gaussian_result(self, ship1, ship2, results, save_path, ozt_line=None):
        if results['method'] == 'kde':
            print(f"The maximum point in the Gaussian KDE is at ({results['max_x']}, {results['max_y']})")
        elif results['method'] == 'gmm':
            print(f"The maximum point in the GMM is at ({results['max_x']}, {results['max_y']})")

        plt.figure(figsize=(10, 5))  # create a new figure

        # Extend the grid for Gaussian computation
        x_min, x_max = results['eCPA_points'][:, 0].min() - 30, results['eCPA_points'][:, 0].max() + 30
        y_min, y_max = results['eCPA_points'][:, 1].min() - 30, results['eCPA_points'][:, 1].max() + 30
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 750), np.linspace(y_min, y_max, 750))
        
        # # Compute Gaussian KDE
        # kde = gaussian_kde(results['eCPA_points'].T)
        # Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        # Compute Gaussian Mixture Model with one component
        gmm = GaussianMixture(n_components=1)
        gmm.fit(results['eCPA_points'])
        Z = np.exp(gmm.score_samples(np.vstack([X.ravel(), Y.ravel()]).T)).reshape(X.shape)
        
        for i in range(1, 3):  # loop to create two subplots
            plt.subplot(1, 2, i)  # create a new subplot
            # Set the limits of the x and y axes
            if i == 1:
                plt.xlim(0, 300)
                plt.ylim(0, 300)
            else:  # zoom in around the maximum point in the second plot
                plt.xlim(results['max_x'] - 15, results['max_x'] + 15)  
                plt.ylim(results['max_y'] - 15, results['max_y'] + 15) 

            # Plot the ships
            plt.plot(ship1[-8]['x'], ship1[-8]['y'], "ob", zorder=2)
            plt.plot(ship2[-8]['x'], ship2[-8]['y'], "og", zorder=2)

            # Plot the ships speed and heading as arrow
            plt.arrow(ship1[-8]['x'], ship1[-8]['y'], ship1[-8]['SOG'] * math.sin(math.radians(ship1[-8]['heading'])),
                    ship1[-8]['SOG'] * math.cos(math.radians(ship1[-8]['heading'])),
                    head_width=5, head_length=5, color='b', zorder=2)
            plt.arrow(ship2[-8]['x'], ship2[-8]['y'], ship2[-8]['SOG'] * math.sin(math.radians(ship2[-8]['heading'])),
                    ship2[-8]['SOG'] * math.cos(math.radians(ship2[-8]['heading'])),
                    head_width=5, head_length=5, color='g', zorder=2)

            # Plot the Gaussian model
            plt.pcolormesh(X, Y, Z, shading='auto', cmap='Blues', alpha=0.7, zorder=1)
            if i == 1:
                plt.scatter(results['eCPA_points'][:, 0], results['eCPA_points'][:, 1], color='blue', marker='.', s=4, zorder=2)
                plt.scatter(results['eCPA_points'][-7][0], results['eCPA_points'][-7][1], color='green', marker='.', s=4, zorder=2)  # the linear eCPA
                plt.scatter(results['eCPA_points'][-6:, 0], results['eCPA_points'][-6:, 1], color='lawngreen', marker='.', s=4, zorder=2)  # the linear eCPA
            else:
                plt.scatter(results['eCPA_points'][:, 0], results['eCPA_points'][:, 1], color='blue', marker='.', zorder=2)
                if ozt_line is not None:
                    plt.scatter(results['eCPA_points'][-7][0], results['eCPA_points'][-7][1], color='green', marker='.', zorder=2)  # the linear eCPA
                    plt.scatter(results['eCPA_points'][-6:, 0], results['eCPA_points'][-6:, 1], color='lawngreen', marker='.', zorder=2)  # the linear eCPA
                else:
                    plt.scatter(results['eCPA_points'][-1][0], results['eCPA_points'][-1][1], color='green', marker='.', zorder=2)
            plt.grid(True, alpha=0.5)

            # Draw a box around the maximum point in the first plot to indicate the zoom area in the second plot
            if i == 1:
                rect = plt.Rectangle((results['max_x'] - 20, results['max_y'] - 20), 40, 40, fill=False, color='black', linestyle='--', alpha=0.5)
                plt.gca().add_patch(rect)

            # Plot the maximum eCPA as red 'X'
            plt.plot(results['max_x'], results['max_y'], "x", markersize=11.5, markeredgewidth=4, color='white', zorder=3)
            plt.plot(results['max_x'], results['max_y'], "xr", markersize=10, markeredgewidth=2, zorder=3)

        plt.savefig(save_path)
        plt.close()


    # def plot_multiple_scenarios(self, scenarios_results, save_path, ship1, ship2, real_cpa):
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.axis(self.axis_measure)
    #     ax.grid(True)
    #     plt.xticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
    #     plt.yticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
    #     plt.xlabel('Nautical Miles [NM]')
    #     plt.ylabel('Nautical Miles [NM]')
    #     # img = plt.imread(self.img_path)
    #     # ax.imshow(img, extent=self.axis_measure)

    #     colors = ['blue', 'green', 'orange', 'purple', 'brown']
    #     legend_handles = []

    #     for i, (scenario, result) in enumerate(scenarios_results.items()):
    #         ship1_color = colors[i % len(colors)]
    #         ship2_color = colors[(i + 1) % len(colors)]

    #         self.plot_ships_and_paths(ax, ship1, ship1_color, f"{scenario} - Own Ship", legend_handles, add_legend=True)
    #         self.plot_ships_and_paths(ax, ship2, ship2_color, f"{scenario} - Target Ship", legend_handles, add_legend=True)

    #         self.plot_predicted_paths(ax, result['predict_position1'], result['predict_speed1'], result['predict_course1'], ship1_color, f"{scenario} - Predicted Own Ship", legend_handles, add_legend=True)
    #         self.plot_predicted_paths(ax, result['predict_position2'], result['predict_speed2'], result['predict_course2'], ship2_color, f"{scenario} - Predicted Target Ship", legend_handles, add_legend=True)

    #         ax.plot(real_cpa[0], real_cpa['1'], "x", markersize=5, markeredgewidth=2, color='red', alpha=0.5, label=f"{scenario} - CPA")

    #     ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=True, framealpha=0.5, edgecolor='black')
    #     fig.savefig(save_path, bbox_inches='tight')
    #     plt.close(fig)

    # def plot_multiple_scenarios(self, scenarios_results, save_path):
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.axis(self.axis_measure)
    #     ax.grid(True)
    #     plt.xticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
    #     plt.yticks([0, 66.15, 132.3, 198.45, 264.6], ['0', '0.5', '1', '1.5', '2'])
    #     plt.xlabel('Nautical Miles [NM]')
    #     plt.ylabel('Nautical Miles [NM]')
    #     # img = plt.imread(self.img_path)
    #     # ax.imshow(img, extent=self.axis_measure)

    #     colors = ['blue', 'green', 'orange', 'purple', 'brown']
    #     legend_handles = []

    #     for i, (scenario, results) in enumerate(scenarios_results.items()):
    #         for _, (pred_method, result) in enumerate(results[0].items()):
    #             ship1_color = colors[i % len(colors)]
    #             ship2_color = colors[(i + 1) % len(colors)]

    #             self.plot_ships_and_paths(ax, results[1], ship1_color, f"{scenario} - Own Ship", legend_handles, add_legend=True)
    #             self.plot_ships_and_paths(ax, results[2], ship2_color, f"{scenario} - Target Ship", legend_handles, add_legend=True)

    #             self.plot_predicted_paths(ax, result['predict_position1'], result['predict_speed1'], result['predict_course1'], ship1_color, f"{scenario} - Predicted Own Ship", legend_handles, add_legend=True)
    #             self.plot_predicted_paths(ax, result['predict_position2'], result['predict_speed2'], result['predict_course2'], ship2_color, f"{scenario} - Predicted Target Ship", legend_handles, add_legend=True)

    #             ax.plot(results[3][0], results[3][1], "x", markersize=5, markeredgewidth=2, color='red', alpha=0.5, label=f"{scenario} - CPA")

    #     ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', frameon=True, framealpha=0.5, edgecolor='black')
    #     fig.savefig(save_path, bbox_inches='tight')
    #     plt.close(fig)