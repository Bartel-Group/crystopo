from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from pymatgen.core import Structure

@dataclass
class StructureType:
    """Configuration for a specific structure type."""
    label: str
    formula_pattern: str
    space_group: int

@dataclass
class BettiData:
    """Container for Betti curves data."""
    label: str
    betti_curves: Tuple[np.ndarray, np.ndarray, np.ndarray]
    structure: Optional[Structure] = None
    mpid: Optional[str] = None
