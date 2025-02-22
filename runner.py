#!/usr/bin/env python
"""
Script for running CrysToPoAnalyzer tasks based on JSON configuration.
Usage: python run_crystopo.py config.json
"""

from typing import Tuple, Optional
from mp_api.client import MPRester
from crystopo import CrysToPoAnalyzer, StructureType, BettiCurvesVisualizer
from pymatgen.core import Structure
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import sys
import os
import re

from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    required_fields = ['task']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration must include '{field}'")

    # Check if MP API key is required and present
    methods_requiring_api = ['mp_ids', 'mp_query']
    if (
        config.get('method') in methods_requiring_api or 
        (config['task'] == 'stability_classification' and config.get('method') != 'local_files')
    ):
        if 'mp_api_key' not in config:
            raise ValueError("Materials Project API key (mp_api_key) is required for this configuration")

    return config

def get_mprester(config: Dict[str, Any]) -> Optional[MPRester]:
    """Get MPRester instance if API key is provided."""
    api_key = config.get('mp_api_key')
    return MPRester(api_key) if api_key else None

def run_betti_calculation(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run Betti curve calculation task."""
    method = config['method']
    visualize = config.get('visualize', False)

    # Setup output directory
    output_base = Path(config.get('output_dir', 'betti_curves'))
    output_base.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    b0_dir = output_base / 'B0'
    b1_dir = output_base / 'B1'
    b2_dir = output_base / 'B2'

    for dir_path in [b0_dir, b1_dir, b2_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    if method in ['mp_ids', 'mp_query']:
        with get_mprester(config) as mpr:
            analyzer.mpr = mpr  # Set MPRester instance
            
            if method == 'mp_query':
                structure_type = StructureType(
                    label=config['label'],
                    formula_pattern=config['formula_pattern'],
                    space_group=config['space_group']
                )
                results = analyzer._fetch_and_process_structures(
                    structure_type.formula_pattern,
                    structure_type.space_group,
                    config.get('sample_size', None)
                )
            else:  # mp_ids
                results = {}
                for mpid in config['mpids']:
                    structure, charge_density = analyzer._fetch_structure_and_density(mpid)
                    if structure is not None and charge_density is not None:
                        results[mpid] = (structure, charge_density)
                    else:
                        print(f"Skipping {mpid} - could not fetch structure or charge density")

    elif method == 'local_files':
        results = {}
        for struct_file, charge_file in zip(
            config['structure_files'],
            config['charge_density_files']
        ):
            structure = Structure.from_file(struct_file)
            with open(charge_file, 'rb') as f:
                charge_density = pickle.load(f)
            results[os.path.basename(struct_file)] = (structure, charge_density)


    # Initialize visualizer if needed
    visualizer = BettiCurvesVisualizer() if visualize else None

    # Calculate and save Betti curves
    for identifier, (structure, charge_density) in results.items():
        try:
            betti0, betti1, betti2 = analyzer.betti_calculator.compute_betti_curves(
                charge_density, structure
            )

            # Save each Betti curve in its respective directory
            np.save(b0_dir / f"{identifier}.npy", betti0)
            np.save(b1_dir / f"{identifier}.npy", betti1)
            np.save(b2_dir / f"{identifier}.npy", betti2)

            print(f"Successfully computed and saved Betti curves for {identifier}")

            # Visualize if requested
            if visualize:
                # Get chemical formula for title
                formula = re.sub(r'(\d+)', r'$_{\1}$', structure.composition.reduced_formula)
                title = f"{formula} ({identifier})"
                visualizer.plot_betti_curves(betti0, betti1, betti2, title=title)
                plt.show()

        except Exception as e:
            print(f"Failed to compute Betti curves for {identifier}: {str(e)}")


def run_structure_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run structure classification task."""
    method = config['method']

    if method == 'mp_query':
        # Create StructureType objects from config
        structure_types = [
            StructureType(
                label=struct['label'],
                formula_pattern=struct['formula_pattern'],
                space_group=struct['space_group']
            )
            for struct in config['structures']
        ]

        # Train classifier
        train_acc, test_acc = analyzer.train_classifier(
            structure_types=structure_types,
            save_path=config['model_path']
        )

    elif method == 'mp_ids':
        # Create dictionary of structures by label
        structures_by_label = {}
        for label, mpids in config['structures'].items():
            structures_data = []
            for mpid in mpids:
                structure, charge_density = analyzer._fetch_structure_and_density(mpid)
                if structure is not None and charge_density is not None:
                    structures_data.append((structure, charge_density))
                else:
                    print(f"Skipping {mpid} - could not fetch structure or charge density")
            structures_by_label[label] = structures_data

        # Train classifier
        train_acc, test_acc = analyzer.train_classifier(
            structures_by_label=structures_by_label,
            save_path=config['model_path']
        )

    elif method == 'local_files':
        # Load structures and charge densities
        structures_by_label = {}
        for label, data in config['data'].items():
            structures = [
                Structure.from_file(path)
                for path in data['structures']
            ]
            charge_densities = [
                pickle.load(open(path, 'rb'))
                for path in data['charge_densities']
            ]
            structures_by_label[label] = list(zip(structures, charge_densities))

        # Train classifier
        train_acc, test_acc = analyzer.train_classifier(
            structures_by_label=structures_by_label,
            save_path=config['model_path']
        )

    elif method == 'betti_curves':
        # Load pre-computed Betti curves
        betti_curves_by_label = {}
        for label, data in config['data'].items():
            curves = []
            for b0_file, b1_file, b2_file in zip(
                data['b0_files'],
                data['b1_files'],
                data['b2_files']
            ):
                curves.append((
                    np.load(b0_file),
                    np.load(b1_file),
                    np.load(b2_file)
                ))
            betti_curves_by_label[label] = curves

        # Train classifier
        train_acc, test_acc = analyzer.train_classifier(
            betti_curves_by_label=betti_curves_by_label,
            save_path=config['model_path']
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_metal_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run metal classification task."""
    method = config['method']

    if method == 'mp_query':
        structure_type = StructureType(
            label=config['label'],
            formula_pattern=config['formula_pattern'],
            space_group=config['space_group']
        )
        structure_types = [structure_type]
        train_acc, test_acc = analyzer.train_classifier(
            structure_types=structure_types,
            property_classifier=True,
            save_path=config['model_path']
        )

    elif method == 'mp_ids':
        # Create dictionary of structures and charge densities from MPIDs
        structures_data = []
        for mpid in config['mpids']:
            structure, charge_density = analyzer._fetch_structure_and_density(mpid)
            if structure is not None and charge_density is not None:
                structures_data.append((structure, charge_density))
            else:
                print(f"Skipping {mpid} - could not fetch structure or charge density")

        train_acc, test_acc = analyzer.train_classifier(
            structures_by_label={'all': structures_data},  # Using a single label since it's property-based
            property_classifier=True,
            save_path=config['model_path']
        )

    elif method == 'local_files':
        # Load structures and charge densities from local files
        structures = [
            Structure.from_file(path)
            for path in config['data']['structures']
        ]
        charge_densities = [
            pickle.load(open(path, 'rb'))
            for path in config['data']['charge_densities']
        ]
        structures_data = list(zip(structures, charge_densities))
        
        train_acc, test_acc = analyzer.train_classifier(
            structures_by_label={'all': structures_data},
            property_classifier=True,
            save_path=config['model_path']
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_stability_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run stability classification task."""
    method = config['method']

    if method == 'mp_ids':
        api_key = config.get('mp_api_key')
        if not api_key:
            raise ValueError("Materials Project API key required for MP method")

        with MPRester(api_key) as mpr:
            stable_structures = []
            unstable_structures = []

            for mpid in config['mpids']:
                try:
                    structure, charge_density = analyzer._fetch_structure_and_density(mpid)
                    mat = mpr.get_structure_by_material_id(mpid)
                    summary = mpr.summary.get_data_by_id(mpid)
                    e_above_hull = summary.energy_above_hull

                    if structure is not None and charge_density is not None and e_above_hull is not None:
                        if e_above_hull < 0.001:  # Using 1 meV threshold
                            stable_structures.append((structure, charge_density))
                        else:
                            unstable_structures.append((structure, charge_density))
                        print(f"Processed {mpid} (E_above_hull = {e_above_hull:.3f} eV/atom)")
                    else:
                        print(f"Skipping {mpid} - missing required data")

                except Exception as e:
                    print(f"Error processing {mpid}: {str(e)}")
                    continue

            structures_by_label = {
                'stable': stable_structures,
                'unstable': unstable_structures
            }

    elif method == 'local_files':
        # Load structures and charge densities from local files
        structures = [
            Structure.from_file(path)
            for path in config['data']['structures']
        ]
        charge_densities = [
            pickle.load(open(path, 'rb'))
            for path in config['data']['charge_densities']
        ]
        
        # Get labels from config
        if 'stability_labels' not in config['data']:
            raise ValueError("Stability labels must be provided for local files method")
        labels = config['data']['stability_labels']

        if len(labels) != len(structures):
            raise ValueError("Number of stability labels must match number of structures")

        # Sort structures into stable/unstable
        stable_structures = []
        unstable_structures = []
        
        for struct, charge, label in zip(structures, charge_densities, labels):
            if label == 0:  # stable
                stable_structures.append((struct, charge))
            else:  # unstable
                unstable_structures.append((struct, charge))

        structures_by_label = {
            'stable': stable_structures,
            'unstable': unstable_structures
        }

    elif method == 'mp_query':
        # Query MP based on formula pattern and space group
        api_key = config.get('mp_api_key')
        if not api_key:
            raise ValueError("Materials Project API key required for MP query method")

        structure_type = StructureType(
            label=config['label'],
            formula_pattern=config['formula_pattern'],
            space_group=config['space_group']
        )

        with MPRester(api_key) as mpr:
            results = analyzer._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                config.get('sample_size', None)
            )

            stable_structures = []
            unstable_structures = []

            for mpid, (structure, charge_density) in results.items():
                try:
                    doc = mpr.materials.summary.get_data_by_id(mpid)
                    e_above_hull = doc.energy_above_hull

                    if e_above_hull is not None:
                        if e_above_hull < 0.001:  # Using 1 meV threshold
                            stable_structures.append((structure, charge_density))
                        else:
                            unstable_structures.append((structure, charge_density))
                        print(f"Processed {mpid} (E_above_hull = {e_above_hull:.3f} eV/atom)")
                except Exception as e:
                    print(f"Error fetching stability data for {mpid}: {str(e)}")
                    continue

            structures_by_label = {
                'stable': stable_structures,
                'unstable': unstable_structures
            }

    else:
        raise ValueError(f"Unknown method: {method}")

    # Train classifier with the collected data
    train_acc, test_acc = analyzer.train_classifier(
        structures_by_label=structures_by_label,
        property_classifier=True,
        save_path=config['model_path']
    )

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_visualization(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run Betti curve visualization task."""
    method = config['method']
    
    if method == 'betti_curves':
        betti_curves = []
        for b0_file, b1_file, b2_file in zip(
            config['b0_files'],
            config['b1_files'],
            config['b2_files']
        ):
            betti_curves.append((
                np.load(b0_file),
                np.load(b1_file),
                np.load(b2_file)
            ))
        
        analyzer.plot_betti_curves(betti_curves=betti_curves)

def run_spectral_embedding(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run spectral embedding visualization task."""
    method = config['method']

    if method == 'mp_query':
        # Get structures and charge densities from MP for each structure type
        betti_curves = []
        color_values = []
        labels = []
        
        for struct_idx, struct_info in enumerate(config['structures']):
            structure_type = StructureType(
                label=struct_info['label'],
                formula_pattern=struct_info['formula_pattern'],
                space_group=struct_info['space_group']
            )
            
            # Fetch structures
            results = analyzer._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                struct_info.get('sample_size', None)
            )
            
            # Compute Betti curves
            for mpid, (structure, charge_density) in results.items():
                if charge_density is not None:
                    betti0, betti1, betti2 = analyzer.betti_calculator.compute_betti_curves(
                        charge_density, structure
                    )
                    betti_curves.append((betti0, betti1, betti2))
                    color_values.append(struct_idx)
                    labels.append(f"{struct_info['label']}: {mpid}")
                    print(f"Processed {mpid} ({struct_info['label']})")
        
        # Create spectral embedding
        analyzer.create_spectral_embedding(
            betti_curves=betti_curves,
            color_values=color_values,
            colorbar_label=config.get('colorbar_label', 'Structure Type')
        )
        plt.show()

    elif method == 'mp_ids':
        # Get structures and charge densities from MP
        betti_curves = []
        labels = []  # to store MPIDs as labels
        
        for mpid in config['mpids']:
            structure, charge_density = analyzer._fetch_structure_and_density(mpid)
            if structure is not None and charge_density is not None:
                betti0, betti1, betti2 = analyzer.betti_calculator.compute_betti_curves(
                    charge_density, structure
                )
                betti_curves.append((betti0, betti1, betti2))
                labels.append(mpid)
                print(f"Processed {mpid}")
            else:
                print(f"Skipping {mpid} - could not fetch structure or charge density")

        # Create color values (just indices if not provided)
        color_values = config.get('color_values', list(range(len(betti_curves))))
        
        # Create spectral embedding
        analyzer.create_spectral_embedding(
            betti_curves=betti_curves,
            color_values=color_values,
            colorbar_label=config.get('colorbar_label', 'Index')
        )
        plt.show()

    elif method == 'betti_curves':
        # Load Betti curves and prepare color values
        betti_curves = []
        color_values = []

        for label_idx, (label, data) in enumerate(config['data'].items()):
            for b0_file, b1_file, b2_file in zip(
                data['b0_files'],
                data['b1_files'],
                data['b2_files']
            ):
                betti_curves.append((
                    np.load(b0_file),
                    np.load(b1_file),
                    np.load(b2_file)
                ))
                color_values.append(label_idx)

        analyzer.create_spectral_embedding(
            betti_curves=betti_curves,
            color_values=color_values,
            colorbar_label=config.get('colorbar_label', 'Type')
        )
        plt.show()
    
    else:
        raise ValueError(f"Unknown method: {method}")

def run_task(config: Dict[str, Any]) -> None:
    """Run the specified task with given configuration."""
    analyzer = CrysToPoAnalyzer()
    
    # Set MPRester for methods that need it
    if config.get('method') in ['mp_ids', 'mp_query']:
        analyzer.mpr = get_mprester(config)
    
    task = config['task']

    if task == 'betti_calculation':
        run_betti_calculation(analyzer, config)
    elif task == 'visualization':
        run_visualization(analyzer, config)
    elif task == 'structure_classification':
        run_structure_classification(analyzer, config)
    elif task == 'metal_classification':
        run_metal_classification(analyzer, config)
    elif task == 'stability_classification':
        run_stability_classification(analyzer, config)
    elif task == 'spectral_embedding':
        run_spectral_embedding(analyzer, config)
    else:
        raise ValueError(f"Unknown task: {task}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_crystopo.py config.json")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_task(config)

if __name__ == '__main__':
    main()
