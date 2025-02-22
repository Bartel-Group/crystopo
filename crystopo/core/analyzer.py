from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from pymatgen.core import Structure

from .config import MPConfig
from .types import StructureType, BettiData
from .calculator import BettiCurvesCalculator
from ..classification.classifier import BettiClassifier
from ..vis.visualizer import BettiCurvesVisualizer
from ..vis.embedding import BettiCurvesEmbedding
from .. import mp_api

class CrysToPoAnalyzer:
    """Main class for crystallographic topology analysis."""

    def __init__(self, config: Optional[MPConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or MPConfig()
        self.betti_calculator = BettiCurvesCalculator()
        random.seed(self.config.random_seed)

    def _fetch_and_process_structures(
        self,
        formula_pattern: str,
        space_group: int,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Helper method to fetch and process structures."""
        mpids = mp_api.find_materials(formula_pattern, space_group, self.config.api_key)
        if sample_size:
            random.shuffle(mpids)
            mpids = mpids[:sample_size]
        return mp_api.download_structure_and_charge_density(mpids, self.config.api_key)

    def _compute_betti_curves(
        self,
        structure_data: Union[Dict, List[Tuple[Structure, np.ndarray]]]
    ) -> List[BettiData]:
        """
        Helper method to compute Betti curves for structures.
        
        Args:
            structure_data: Either:
                - Dictionary from MP API (mpid -> (structure, charge_density))
                - List of (structure, charge_density) tuples from user
        """
        betti_data = []
        
        if isinstance(structure_data, dict):
            # MP API data
            for mpid, (structure, charge_density) in structure_data.items():
                if charge_density is not None:
                    curves = self.betti_calculator.compute_betti_curves(charge_density, structure)
                    betti_data.append(BettiData(
                        label='',  # Will be set by calling method
                        betti_curves=curves,
                        structure=structure,
                        mpid=mpid
                    ))
        else:
            # User-provided structures and charge densities
            for structure, charge_density in structure_data:
                curves = self.betti_calculator.compute_betti_curves(charge_density, structure)
                betti_data.append(BettiData(
                    label='',  # Will be set by calling method
                    betti_curves=curves,
                    structure=structure,
                    mpid=None
                ))
                
        return betti_data

    def _fetch_structure_and_density(
        self,
        mpid: str
    ) -> Tuple[Optional[Structure], Optional[np.ndarray]]:
        """
        Fetch a single structure and its charge density from Materials Project.
        
        Args:
            mpid: Materials Project ID
            
        Returns:
            Tuple of (structure, charge_density), either may be None if fetch fails
        """
        try:
            results = mp_api.download_structure_and_charge_density(
                [mpid],  # Pass as list since the API expects a list
                self.config.api_key
            )
            if mpid in results:
                return results[mpid]
            return None, None
        except Exception as e:
            print(f"Failed to fetch {mpid}: {str(e)}")
            return None, None

    def train_classifier(
        self,
        save_path: str,
        property_classifier: bool = False,
        structure_types: Optional[List[StructureType]] = None,
        structures_by_label: Optional[Dict[str, List[Tuple[Structure, Optional[str]]]]] = None,
        betti_curves_by_label: Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None,
        sample_size: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Train a classifier using either MP API data, provided structures, or provided Betti curves.

        Args:
            save_path: Path to save the trained classifier
            property_classifier: If True, classifies based on material properties
            structure_types: List of StructureType objects for MP API fetching
            structures_by_label: Dict mapping labels to lists of (structure, mpid) tuples
            betti_curves_by_label: Dict mapping labels to lists of pre-computed Betti curves
            sample_size: Number of samples per class (for MP API fetching)

        Returns:
            Tuple of (training_accuracy, test_accuracy)
        """
        data = {}
        
        if betti_curves_by_label:
            # Use provided Betti curves directly
            data = {label: curves for label, curves in betti_curves_by_label.items()}
            
        elif structures_by_label:
            # Compute Betti curves from provided structures
            for label, structures in structures_by_label.items():
                betti_data = self._compute_betti_curves(structures)
                data[label] = [bd.betti_curves for bd in betti_data]
                
        elif structure_types:
            # Fetch from MP API
            for struct_type in structure_types:
                results = self._fetch_and_process_structures(
                    struct_type.formula_pattern,
                    struct_type.space_group,
                    sample_size
                )

                if property_classifier:
                    # Initialize containers for property-based classification
                    data['metal'] = []
                    data['insulator'] = []

                    for mpid, (structure, charge_density) in results.items():
                        if charge_density is not None:
                            betti0, betti1, betti2 = self.betti_calculator.compute_betti_curves(charge_density, structure)
                            band_gap = mp_api.download_band_gap(mpid, self.config.api_key)
                            category = 'metal' if band_gap == 0 else 'insulator'
                            data[category].append((betti0, betti1, betti2))
                else:
                    # Structure-based classification
                    betti_data = self._compute_betti_curves(results)
                    data[struct_type.label] = [bd.betti_curves for bd in betti_data]
        
        else:
            raise ValueError("Must provide one of: structure_types, structures_by_label, or betti_curves_by_label")

        # Train and save classifier
        classifier = BettiClassifier(n_estimators=100, random_state=42)
        train_acc, test_acc = classifier.train(data)
        classifier.save(save_path)

        return train_acc, test_acc

    def apply_classifier(
        self,
        model_path: str,
        property_classifier: bool = False,
        structure_type: Optional[StructureType] = None,
        structures: Optional[List[Tuple[Structure, Optional[str]]]] = None,
        betti_curves: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        sample_size: Optional[int] = None,
    ) -> List[Dict]:
        """
        Apply a trained classifier to new data.

        Args:
            model_path: Path to the trained classifier
            property_classifier: If True, includes property-based ground truth
            structure_type: StructureType object for MP API fetching
            structures: List of (structure, mpid) tuples
            betti_curves: List of pre-computed Betti curves
            sample_size: Number of samples (for MP API fetching)
        """
        classifier = BettiClassifier.load(model_path)
        predictions = []

        if betti_curves is not None:
            # Use provided Betti curves directly
            for curves in betti_curves:
                pred_info = {
                    'prediction': classifier.predict(curves),
                    'probabilities': classifier.predict_proba(curves)
                }
                predictions.append(pred_info)

        elif structures is not None:
            # Compute Betti curves from provided structures
            betti_data = self._compute_betti_curves(structures)
            for bd in betti_data:
                pred_info = {
                    'mpid': bd.mpid,
                    'prediction': classifier.predict(bd.betti_curves),
                    'probabilities': classifier.predict_proba(bd.betti_curves)
                }
                predictions.append(pred_info)

        elif structure_type is not None:
            # Fetch from MP API
            results = self._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                sample_size
            )
            
            betti_data = self._compute_betti_curves(results)
            for bd in betti_data:
                pred_info = {
                    'mpid': bd.mpid,
                    'prediction': classifier.predict(bd.betti_curves),
                    'probabilities': classifier.predict_proba(bd.betti_curves)
                }
                
                if property_classifier and bd.mpid:
                    band_gap = mp_api.download_band_gap(bd.mpid, self.config.api_key)
                    pred_info['ground_truth'] = 'metal' if band_gap == 0 else 'insulator'
                
                predictions.append(pred_info)

        else:
            raise ValueError("Must provide one of: structure_type, structures, or betti_curves")

        return predictions

    def plot_betti_curves(
        self,
        structure_type: Optional[StructureType] = None,
        structures: Optional[List[Tuple[Structure, Optional[str]]]] = None,
        betti_curves: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        sample_size: Optional[int] = None
    ) -> None:
        """Plot Betti curves from various data sources."""
        visualizer = BettiCurvesVisualizer()

        if betti_curves is not None:
            for i, (betti0, betti1, betti2) in enumerate(betti_curves):
                fig = visualizer.plot_betti_curves(betti0, betti1, betti2, title=f"Curve {i+1}")
                plt.show()

        elif structures is not None:
            betti_data = self._compute_betti_curves(structures)
            for bd in betti_data:
                title = f"Structure {bd.mpid}" if bd.mpid else "Structure"
                if bd.structure:
                    title = re.sub(r'(\d+)', r'$_{\1}$', bd.structure.composition.reduced_formula)
                fig = visualizer.plot_betti_curves(*bd.betti_curves, title=title)
                plt.show()

        elif structure_type is not None:
            results = self._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                sample_size
            )
            betti_data = self._compute_betti_curves(results)
            for bd in betti_data:
                title = re.sub(r'(\d+)', r'$_{\1}$', bd.structure.composition.reduced_formula)
                fig = visualizer.plot_betti_curves(*bd.betti_curves, title=title)
                plt.show()

        else:
            raise ValueError("Must provide one of: structure_type, structures, or betti_curves")

    def create_spectral_embedding(
        self,
        structure_type: Optional[StructureType] = None,
        structures: Optional[List[Tuple[Structure, Optional[str]]]] = None,
        betti_curves: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
        sample_size: Optional[int] = None,
        color_values: Optional[List] = None,
        colorbar_label: Optional[str] = None
    ) -> None:
        """Create spectral embedding visualization from various data sources."""
        labels = []
        curves_list = []

        if betti_curves is not None:
            curves_list = betti_curves
            labels = [f"Curve {i+1}" for i in range(len(betti_curves))]

        elif structures is not None:
            betti_data = self._compute_betti_curves(structures)
            for bd in betti_data:
                curves_list.append(bd.betti_curves)
                labels.append(bd.mpid if bd.mpid else "Structure")

        elif structure_type is not None:
            results = self._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                sample_size
            )
            betti_data = self._compute_betti_curves(results)
            for bd in betti_data:
                curves_list.append(bd.betti_curves)
                labels.append(bd.mpid)

        else:
            raise ValueError("Must provide one of: structure_type, structures, or betti_curves")

        embedder = BettiCurvesEmbedding(method='spectral')
        embedding = embedder.fit_transform(curves_list)
        
        embedder.plot_embedding(
            color_values=color_values,
            labels=labels,
            title="Betti Curves Embedding",
            colorbar_label=colorbar_label
        )
        plt.show()
