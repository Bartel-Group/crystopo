from typing import Tuple, List, Optional
from numpy.typing import NDArray
from pymatgen.core import Structure
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import numpy as np
import gudhi as gd

class BettiCurvesCalculator:
    def __init__(self,
                 num_filtrations: int = 500,
                 filtration_range: Tuple[float, float] = (0, 2),
                 show_progress: bool = True,
                 smooth_sigma: Optional[float] = 2.0,
                 betti_cap: float = 5.0) -> None:
        """
        Initialize the calculator with parameters for filtration.
        Now uses superlevel set filtration (density >= threshold).

        Args:
            num_filtrations: Number of filtration points
            filtration_range: Range of filtration values (min, max)
                            Default changed to (0, 2) to go from low to high
            show_progress: Whether to show progress bar during computation
            smooth_sigma: Standard deviation for Gaussian smoothing.
                        If None, no smoothing is applied.
            betti_cap: Maximum allowed value for Betti numbers (default: 5.0).
                      Any values exceeding this will be set to this value.
        """
        self.num_filtrations = num_filtrations
        self.filtration_range = filtration_range
        self.show_progress = show_progress
        self.smooth_sigma = smooth_sigma
        self.betti_cap = betti_cap
        self.filtrations: NDArray[np.float64] = np.linspace(
            filtration_range[0],
            filtration_range[1],
            num_filtrations
        )

    def _compute_persistence_diagram(self,
                                  charge_density: NDArray[np.float64]
                                  ) -> gd.CubicalComplex:
        """
        Compute the persistence diagram using gudhi.
        Now negates the charge density to convert superlevel to sublevel filtration.

        Args:
            charge_density: 3D array of charge density values

        Returns:
            Computed cubical complex
        """
        # Negate charge density to convert superlevel to sublevel filtration
        negated_density = -charge_density
        cubical_complex = gd.CubicalComplex(top_dimensional_cells=negated_density)
        cubical_complex.persistence()
        return cubical_complex

    def _compute_betti_numbers(self,
                             cubical_complex: gd.CubicalComplex,
                             filtration_value: float
                             ) -> List[int]:
        """
        Compute Betti numbers for a specific filtration value.
        Note: The filtration value is negated internally since we negated the density.

        Args:
            cubical_complex: Computed gudhi cubical complex
            filtration_value: Value at which to compute Betti numbers

        Returns:
            Betti numbers [b0, b1, b2]
        """
        epsilon: float = 0.005  # Small window around filtration value
        # Negate the filtration value since we negated the density
        negated_value = -filtration_value
        return cubical_complex.persistent_betti_numbers(
            negated_value - epsilon,
            negated_value + epsilon
        )

    # Rest of the methods remain unchanged
    def _determine_supercell_size(self, structure: Structure) -> int:
        """
        Determine supercell size based on number of atoms.

        Args:
            structure: Pymatgen Structure object

        Returns:
            Supercell size (1, 2, or 3)
        """
        n_atoms = len(structure)
        if n_atoms < 10:
            return 3
        elif n_atoms < 50:
            return 2
        else:
            return 1

    def _smooth_curves(self,
                      curves: Tuple[NDArray[np.float64],
                                  NDArray[np.float64],
                                  NDArray[np.float64]],
                      sigma: float
                      ) -> Tuple[NDArray[np.float64],
                                NDArray[np.float64],
                                NDArray[np.float64]]:
        """
        Apply Gaussian smoothing to Betti curves.

        Args:
            curves: Tuple of three Betti curves
            sigma: Standard deviation for Gaussian kernel

        Returns:
            Smoothed Betti curves
        """
        return tuple(gaussian_filter1d(curve, sigma) for curve in curves)

    def compute_betti_curves(self,
                           charge_density: NDArray[np.float64],
                           structure: Structure,
                           smooth_sigma: Optional[float] = None
                           ) -> Tuple[NDArray[np.float64],
                                    NDArray[np.float64],
                                    NDArray[np.float64]]:
        """
        Compute normalized Betti curves from charge density using superlevel filtration.
        Now analyzes regions where density >= threshold, iterating from low to high density.
        
        Automatically determines supercell size based on number of atoms:
        - n_atoms < 10: supercell_size = 3
        - 10 <= n_atoms < 50: supercell_size = 2
        - n_atoms >= 50: supercell_size = 1

        Args:
            charge_density: 3D array of charge density values
            structure: Pymatgen Structure object
            smooth_sigma: Optional override for smoothing parameter.
                        If None, uses the value set during initialization.

        Returns:
            Three numpy arrays (betti0s, betti1s, betti2s) containing the
            normalized Betti curves
        """
        # First normalize by primitive cell volume
        primitive_volume = structure.volume
        charge_density = charge_density / primitive_volume

        # Determine and create supercell
        supercell_size: int = self._determine_supercell_size(structure)
        if supercell_size > 1:
            structure.make_supercell(supercell_size)
            charge_density = np.tile(
                charge_density,
                (supercell_size, supercell_size, supercell_size)
            )

        # Get number of atoms in supercell for normalization
        n_atoms = len(structure)

        # Initialize arrays for Betti curves
        betti0s: NDArray[np.float64] = np.zeros(self.num_filtrations)
        betti1s: NDArray[np.float64] = np.zeros(self.num_filtrations)
        betti2s: NDArray[np.float64] = np.zeros(self.num_filtrations)

        # Create progress bar
        if self.show_progress:
            print("Computing persistence diagram...")

        # Compute persistence diagram (using negated density)
        cubical_complex: gd.CubicalComplex = self._compute_persistence_diagram(charge_density)

        # Compute Betti curves with progress bar
        iterator = enumerate(self.filtrations)
        if self.show_progress:
            iterator = tqdm(iterator,
                          total=self.num_filtrations,
                          desc="Computing Betti curves",
                          unit="filtration")

        for j, filtration_value in iterator:
            betti_numbers: List[int] = self._compute_betti_numbers(
                cubical_complex,
                filtration_value
            )

            # Normalize by number of atoms in supercell
            betti0s[j] = betti_numbers[0] / n_atoms
            betti1s[j] = betti_numbers[1] / n_atoms
            betti2s[j] = betti_numbers[2] / n_atoms

        # Apply capping to avoid irregularities
        betti0s = np.minimum(betti0s, self.betti_cap)
        betti1s = np.minimum(betti1s, self.betti_cap)
        betti2s = np.minimum(betti2s, self.betti_cap)

        # Apply smoothing if requested
        sigma = smooth_sigma if smooth_sigma is not None else self.smooth_sigma
        if sigma is not None:
            betti0s, betti1s, betti2s = self._smooth_curves(
                (betti0s, betti1s, betti2s),
                sigma
            )

        return betti0s, betti1s, betti2s
