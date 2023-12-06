import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd

class Visualizer:
    def __init__(self):
        # Set default font size for readability
        plt.rcParams.update({'font.size': 11})
        # Set a colorblind-friendly palette as default
        self.palette = sns.color_palette("colorblind")
        # Set default line width for clarity
        self.line_width = 2

    def create_colorheatmap(self, weights, weights_2=None, column_labels=None, column_labels_2=None):
        """
        Creates and displays one or two color-blind friendly heatmaps, sorted alphabetically by column labels.
        If weights_2 is provided, it displays it as a second subplot.
        column_labels and column_labels_2 are lists of labels for the columns of weights and weights_2, respectively.
        """
        # Convert weights to DataFrame for easier manipulation
        df_weights = pd.DataFrame(weights, columns=column_labels)

        # Sort the DataFrame by column labels
        df_weights = df_weights.reindex(sorted(df_weights.columns), axis=1)

        if weights_2 is None:
            # Only one heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_weights, cmap='cividis', cbar=True)
            plt.title("Heatmap of weights over months")
            plt.xlabel("Assets")
            plt.ylabel("Months")
        else:
            # Convert weights_2 to DataFrame and sort
            df_weights_2 = pd.DataFrame(weights_2, columns=column_labels_2)
            df_weights_2 = df_weights_2.reindex(sorted(df_weights_2.columns), axis=1)

            # Two heatmaps as subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

            sns.heatmap(df_weights, cmap='cividis', cbar=True, ax=ax1)
            ax1.set_title("Heatmap of Sector weights over months")
            ax1.set_xlabel("Assets")
            ax1.set_ylabel("Months")

            sns.heatmap(df_weights_2, cmap='cividis', cbar=True, ax=ax2)
            ax2.set_title("Heatmap of Industry Group weights over months")
            ax2.set_xlabel("Assets")
            ax2.set_ylabel("Months")

        return fig

    def plot_strategies_with_benchmark(self, dates, strategy_returns_lists, benchmark_returns, labels, start_month, strategy_returns_lists_2=None, labels_2=None):
        """
        Plots multiple strategy returns along with a benchmark return.
        Handles both single and multiple strategy return series.
        """
        # Convert single series to list for consistency
        if isinstance(strategy_returns_lists, np.ndarray):
            strategy_returns_lists = [strategy_returns_lists]
            labels = [labels]

        if strategy_returns_lists_2 is not None:
            if isinstance(strategy_returns_lists_2, np.ndarray):
                strategy_returns_lists_2 = [strategy_returns_lists_2]
                labels_2 = [labels_2]

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each strategy return series in the first list
        for i, returns in enumerate(strategy_returns_lists):
            cumulative_returns = np.cumprod(1 + returns[start_month:]) - 1
            ax.plot(dates[start_month:], cumulative_returns, label=labels[i], color=self.palette[i % len(self.palette)], linestyle='-', linewidth=self.line_width)

        # Plot each strategy return series in the second list if provided
        if strategy_returns_lists_2 is not None:
            for i, returns in enumerate(strategy_returns_lists_2):
                cumulative_returns = np.cumprod(1 + returns[start_month:]) - 1
                ax.plot(dates[start_month:], cumulative_returns, label=labels_2[i], color=self.palette[(len(strategy_returns_lists) + i) % len(self.palette)], linestyle='-', linewidth=self.line_width)

        # Plot the benchmark return series
        cumulative_benchmark_returns = np.cumprod(1 + benchmark_returns[start_month:].squeeze()) - 1
        ax.plot(dates[start_month:], cumulative_benchmark_returns, label='Benchmark', color='darkblue', linestyle='-', linewidth=self.line_width)

        # Add labels, title, and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title('Strategies vs. Benchmark Cumulative Returns')
        ax.legend()

        # Add gridlines for horizontal and vertical axes
        ax.grid(axis='y')
        ax.grid(axis='x')

        # Format the date for monthly data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Display the plot
        plt.tight_layout()
        return fig

    def plot_robustness_check(self, check_range, sharpe_ratios, check_type, sharpe_ratios_2=None, labels=None):
        """
        Plots the results of a robustness check.
        Optionally plots a second line if sharpe_ratios_2 is provided.
        """
        plt.figure()
        if sharpe_ratios_2 is None or labels is None:
            # Plot only one line without labels
            plt.plot(check_range, sharpe_ratios, linewidth=self.line_width, color=self.palette[0])
        else:
            # Plot two lines with labels
            plt.plot(check_range, sharpe_ratios, linewidth=self.line_width, color=self.palette[0], label=labels[0])
            plt.plot(check_range, sharpe_ratios_2, linewidth=self.line_width, color=self.palette[1], label=labels[1])
            plt.legend()

        plt.xlabel(check_type.replace("_", " "))
        plt.ylabel("Sharpe Ratio")
        plt.title(f"Robustness Check: {check_type.replace('_', ' ').title()}")

        # Add gridlines for horizontal and vertical axes
        plt.grid(axis='y')
        plt.grid(axis='x')
        
        return plt
