from typing import Optional, Tuple, Sequence, List, Union
from sklearn.manifold import SpectralEmbedding, TSNE
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import mplcursors
import umap

class BettiCurvesEmbedding:
    """Class for dimensionality reduction and visualization of Betti curves."""
    
    def __init__(self,
                 method: str = 'spectral',
                 random_state: Optional[int] = None):
        """
        Initialize the embedding visualizer.
        
        Args:
            method: Embedding method ('spectral', 'tsne', or 'umap')
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state
        self.embedding = None
        
        # Initialize the chosen embedding method
        if self.method == 'spectral':
            self.reducer = SpectralEmbedding(
                n_components=2,
                random_state=random_state
            )
        elif self.method == 'tsne':
            self.reducer = TSNE(
                n_components=2,
                random_state=random_state
            )
        elif self.method == 'umap':
            self.reducer = umap.UMAP(
                n_components=2,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'spectral', 'tsne', or 'umap'")
    
    def fit_transform(self,
                     betti_curves: Sequence[Tuple[NDArray[np.float64], 
                                                NDArray[np.float64], 
                                                NDArray[np.float64]]]
                     ) -> NDArray[np.float64]:
        """
        Fit and transform the Betti curves to 2D embeddings.
        
        Args:
            betti_curves: Sequence of tuples, each containing (β₀, β₁, β₂) curves
            
        Returns:
            2D array of embedded coordinates
        """
        # Concatenate curves for each sample
        X = np.array([np.concatenate(curves) for curves in betti_curves])
        
        # Compute embedding
        self.embedding = self.reducer.fit_transform(X)
        return self.embedding
    
    def plot_embedding(self,
                      color_values: Optional[Union[NDArray[np.float64], List[float]]] = None,
                      labels: Optional[List[str]] = None,
                      title: str = "",
                      colorbar_label: str = "",
                      figsize: Tuple[int, int] = (10, 8),
                      cmap: str = 'viridis',
                      alpha: float = 0.8,
                      s: float = 100,
                      save_path: Optional[str] = None,
                      dpi: int = 300) -> plt.Figure:
        """
        Plot the embedding with optional coloring and labels.
        
        Args:
            color_values: Values to use for coloring points
            labels: Labels for hover annotations
            title: Plot title
            colorbar_label: Label for colorbar
            figsize: Figure size in inches
            cmap: Colormap for points
            alpha: Transparency of points
            s: Size of points
            save_path: Path to save figure
            dpi: DPI for saved figure
            
        Returns:
            matplotlib Figure object
        """
        if self.embedding is None:
            raise ValueError("Must call fit_transform before plotting")
            
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Create scatter plot
        if color_values is not None:
            sc = plt.scatter(self.embedding[:, 0], self.embedding[:, 1],
                           c=color_values, cmap=cmap, alpha=alpha, s=s)
            # Add colorbar
            cbar = plt.colorbar(sc)
            cbar.set_label(colorbar_label, size=20, labelpad=16)
            cbar.ax.tick_params(labelsize=17)
        else:
            sc = plt.scatter(self.embedding[:, 0], self.embedding[:, 1],
                           alpha=alpha, s=s)
        
        # Add hover labels if provided
        if labels is not None:
            cursor = mplcursors.cursor(sc, hover=True)
            
            @cursor.connect("add")
            def on_add(sel):
                """Function/Class documentation here."""
                sel.annotation.set(
                    text=labels[sel.target.index],
                    position=(0.001, 0.001)
                )
                sel.annotation.xy = (
                    self.embedding[sel.target.index, 0],
                    self.embedding[sel.target.index, 1]
                )
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.6)
        
        # Customize plot
        plt.title(title, fontsize=24, pad=20)
        
        # Remove axes for cleaner look
        plt.xticks([])
        plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
