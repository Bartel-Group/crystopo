from typing import Optional, Tuple, Sequence, List, Union
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplcursors

class BettiCurvesVisualizer:
    def __init__(self,
                 figsize: Tuple[int, int] = (10, 6),
                 colors: Optional[Sequence[str]] = None):
        """
        Initialize the visualizer with plotting parameters.

        Args:
            figsize: Figure size (width, height) in inches
            colors: Custom colors for each Betti curve. If None, uses default color scheme
        """
        self.figsize = figsize
        self.colors = colors or ['purple', 'darkorange', 'green']  # Default colors

    def plot_betti_curves(self,
                         betti0: NDArray[np.float64],
                         betti1: NDArray[np.float64],
                         betti2: NDArray[np.float64],
                         filtration_values: Optional[NDArray[np.float64]] = None,
                         x_range: tuple = (0, 2),
                         title: str = "Betti Curves",
                         xlabel: str = r"$\rho_{\mathrm{min}}$ (e/Å$^{3}$)",
                         ylabel: str = "Betti Number",
                         legend_loc: str = "upper right",
                         grid: bool = True,
                         save_path: Optional[str] = None,
                         dpi: int = 400) -> None:
        """
        Plot all three Betti curves on a single figure.

        Args:
            betti0: First Betti curve (β₀)
            betti1: Second Betti curve (β₁)
            betti2: Third Betti curve (β₂)
            filtration_values: X-axis values. If None, uses array indices
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_loc: Location of the legend
            grid: Whether to show grid
            save_path: If provided, saves the plot to this path
            dpi: DPI for saved figure
        """

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Generate x-values if not provided
        if filtration_values is None:
            filtration_values = np.arange(len(betti0))

        # Normalize filtration values to x-range
        filtration_values = [(x - min(filtration_values)) * (x_range[1] - x_range[0]) /
            (max(filtration_values) - min(filtration_values)) + x_range[0] for x in filtration_values]

        # Plot curves
        curves = [
            (betti0, "β₀"),
            (betti1, "β₁"),
            (betti2, "β₂")
        ]

        for (curve, symbol), color in zip(curves, self.colors):
            ax.plot(filtration_values,
                   curve,
                   label=f"{symbol}",
                   color=color,
                   linewidth=2)

        # Customize plot
        ax.set_title(title, pad=10, fontsize=24)
        ax.set_xlabel(xlabel, fontsize=24, labelpad=12)
        ax.set_ylabel(ylabel, fontsize=24, labelpad=16)
        ax.tick_params(axis='both', which='major', labelsize=21)
        ax.set_xlim(x_range[0], x_range[1])

        if grid:
            ax.grid(True, alpha=0.5)

        # Add legend
        ax.legend(loc=legend_loc,
                 frameon=True,
                 fancybox=True,
                 shadow=True,
                 fontsize=24)

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig
