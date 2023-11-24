import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        # Initialization can be used to set up any necessary properties or default values
        pass

    def create_colorheatmap(self, weights):
        """
        Creates and displays a color-blind friendly heatmap.

        Parameters:
        - weights (2D array-like): The data to be visualized in the heatmap.
        """
        plt.figure(figsize=(10, 6))
        # Using the 'cividis' colormap which is designed to be colorblind-friendly
        sns.heatmap(weights, cmap='cividis', cbar=True)
        plt.title("Heatmap of weights over months")
        plt.xlabel("Assets")
        plt.ylabel("Months")
        plt.show()

    def plot_strategies_with_benchmark(self, dates, strategy_returns_list, benchmark_returns, labels, start_month):
        """
        Plots strategy returns (single or multiple) along with a benchmark return in a color-blind friendly way.

        Parameters:
        - dates (array-like): Dates for the x-axis.
        - strategy_returns_list (array-like or list of array-like): Total returns of one or multiple strategies.
        - benchmark_returns (array-like): Total returns of the benchmark.
        - labels (list of str): List of labels for the strategy return series.
        - start_month (int): The starting index for the plot.
        """
        # Check if strategy_returns_list is a single series or a list of series
        if isinstance(strategy_returns_list, np.ndarray):
            strategy_returns_list = [strategy_returns_list]  # Convert to a list for consistency
            labels = [labels]  # Assume labels is a single string and convert to a list

        if len(strategy_returns_list) > 5:
            raise ValueError("Cannot plot more than 5 strategy return series.")

        # Define a list of color-blind friendly colors for strategies
        strategy_colors = ['seagreen', 'darkorange', 'purple', 'grey', 'brown']

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each strategy return series
        for i, returns in enumerate(strategy_returns_list):
            cumulative_returns = np.cumprod(1 + returns[start_month:]) - 1
            ax.plot(dates[start_month:], cumulative_returns, label=labels[i], color=strategy_colors[i], linestyle='-')

        # Plot the benchmark return series
        cumulative_benchmark_returns = np.cumprod(1 + benchmark_returns[start_month:].squeeze()) - 1
        ax.plot(dates[start_month:], cumulative_benchmark_returns, label='Benchmark', color='darkblue', linestyle='--')

        # Add labels, title, and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Strategies vs. Benchmark Cumulative Returns')
        ax.legend()

        # Format the date for monthly data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Display the plot
        plt.tight_layout()
        plt.show()

    def plot_robustness_check(self, check_range, sharp_ratios, check_type, line_width=2, color='blue'):
        """
        Plots the results of a robustness check.

        Parameters:
        - check_range (range): The range of values used in the robustness check.
        - sharp_ratios (list or array-like): The calculated Sharp Ratios for each value in the check range.
        - check_type (str): The type of robustness check ('lookback', 'investment_horizon', 'holdings').
        - line_width (int, optional): The width of the line in the plot. Default is 2.
        - color (str, optional): The color of the line in the plot. Default is 'blue'.
        """
        plt.figure()
        plt.plot(check_range, sharp_ratios, linewidth=line_width, color=color)
        plt.xlabel(check_type.replace("_", " "))
        plt.ylabel("Sharpe Ratio")
        plt.title(f"Robustness Check: {check_type.replace('_', ' ').title()}")
        plt.show()