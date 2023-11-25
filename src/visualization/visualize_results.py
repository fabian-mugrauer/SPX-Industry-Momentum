import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        # Set default font size for readability
        plt.rcParams.update({'font.size': 11})
        # Set a colorblind-friendly palette as default
        self.palette = sns.color_palette("colorblind")
        # Set default line width for clarity
        self.line_width = 2

    def create_colorheatmap(self, weights, weights_2=None):
        """
        Creates and displays one or two color-blind friendly heatmaps.
        If weights_2 is provided, it displays it as a second subplot.
        """
        if weights_2 is None:
            # Only one heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(weights, cmap='cividis', cbar=True)
            plt.title("Heatmap of weights over months")
            plt.xlabel("Assets")
            plt.ylabel("Months")
        else:
            # Two heatmaps as subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

            sns.heatmap(weights, cmap='cividis', cbar=True, ax=ax1)
            ax1.set_title("Heatmap of weights over months")
            ax1.set_xlabel("Assets")
            ax1.set_ylabel("Months")

            sns.heatmap(weights_2, cmap='cividis', cbar=True, ax=ax2)
            ax2.set_title("Heatmap of second weights over months")
            ax2.set_xlabel("Assets")
            ax2.set_ylabel("Months")

        plt.show()

    def plot_strategies_with_benchmark(self, dates, strategy_returns_list, benchmark_returns, labels, start_month, strategy_returns_list_2=None, label_2='Industry groups'):
        """
        Plots strategy returns along with a benchmark return.
        Optionally plots a second strategy line if strategy_returns_list_2 is provided.
        """
        # Check if strategy_returns_list is a single series or a list of series
        if isinstance(strategy_returns_list, np.ndarray):
            strategy_returns_list = [strategy_returns_list]  # Convert to a list for consistency
            labels = [labels]  # Assume labels is a single string and convert to a list

        if len(strategy_returns_list) > 5:
            raise ValueError("Cannot plot more than 5 strategy return series.")

        # Use the default palette for strategy colors
        strategy_colors = self.palette[:len(strategy_returns_list)]

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each strategy return series
        for i, returns in enumerate(strategy_returns_list):
            cumulative_returns = np.cumprod(1 + returns[start_month:]) - 1
            ax.plot(dates[start_month:], cumulative_returns, label=labels[i], color=strategy_colors[i], linestyle='-', linewidth=self.line_width)

        # Plot the second strategy return series if provided
        if strategy_returns_list_2 is not None:
            cumulative_returns_2 = np.cumprod(1 + strategy_returns_list_2[start_month:]) - 1
            ax.plot(dates[start_month:], cumulative_returns_2, label=label_2, color='orange', linestyle='-', linewidth=self.line_width)

        # Plot the benchmark return series
        cumulative_benchmark_returns = np.cumprod(1 + benchmark_returns[start_month:].squeeze()) - 1
        ax.plot(dates[start_month:], cumulative_benchmark_returns, label='Benchmark', color='darkblue', linestyle='--', linewidth=self.line_width)

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

    def plot_robustness_check(self, check_range, sharp_ratios, check_type):
        """
        Plots the results of a robustness check.
        """
        plt.figure()
        plt.plot(check_range, sharp_ratios, linewidth=self.line_width, color='blue')
        plt.xlabel(check_type.replace("_", " "))
        plt.ylabel("Sharpe Ratio")
        plt.title(f"Robustness Check: {check_type.replace('_', ' ').title()}")
        plt.show()
