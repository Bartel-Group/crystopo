"""
Benchmarks using non-topological descriptors
Usage: python benchmarks.py config.json
"""

from crystopo import CrysToPoAnalyzer, StructureType
from crystopo import mp_api
from crystopo.classification import BettiClassifier
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import sys
import os
import pkg_resources
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Import descriptor generators
from chgnet.model import CHGNet
from skipatom import SkipAtomInducedModel, sum_pool
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.site import CrystalNNFingerprint
from mace.calculators import MACECalculator
from dscribe.descriptors import SOAP

def get_default_skipatom_paths():
    """Get default paths for SkipAtom model and training data relative to package."""
    try:
        # Get the installation directory of the crystopo package
        crystopo_path = pkg_resources.resource_filename('crystopo', '')
        data_dir = os.path.join(crystopo_path, 'data')

        model_path = os.path.join(data_dir, 'mp_2020_10_09.dim200.model')
        data_path = os.path.join(data_dir, 'mp_2020_10_09.training.data')

        # Check if files exist
        if os.path.exists(model_path) and os.path.exists(data_path):
            return model_path, data_path

        # If not in data dir, try the package root
        model_path = os.path.join(crystopo_path, 'mp_2020_10_09.dim200.model')
        data_path = os.path.join(crystopo_path, 'mp_2020_10_09.training.data')

        if os.path.exists(model_path) and os.path.exists(data_path):
            return model_path, data_path

        print(f"Warning: SkipAtom files not found in {data_dir} or {crystopo_path}")
        return None, None

    except Exception as e:
        print(f"Warning: Error locating default SkipAtom files: {e}")
        return None, None

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Required fields for all tasks
    required_fields = ['task', 'method', 'descriptor_type']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration must include '{field}'")

    # Check if MP API key is required and present
    methods_requiring_api = ['mp_query', 'mp_ids']
    if config['method'] in methods_requiring_api:
        if 'mp_api_key' not in config:
            raise ValueError("Materials Project API key (mp_api_key) is required for MP-based methods")

    # Validate descriptor type
    valid_descriptors = ['mace', 'chgnet', 'matminer', 'soap', 'skipatom']
    if config['descriptor_type'] not in valid_descriptors:
        raise ValueError(f"descriptor_type must be one of {valid_descriptors}")

    # Validate method
    valid_methods = ['mp_query', 'mp_ids', 'local_files']
    if config['method'] not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Set default SkipAtom paths if not provided
    if config['descriptor_type'] == 'skipatom':
        if 'skipatom_model_path' not in config or 'skipatom_training_data_path' not in config:
            default_model_path, default_data_path = get_default_skipatom_paths()
            if default_model_path is None or default_data_path is None:
                raise ValueError(
                    "Could not find default SkipAtom files and no paths provided in config.\n"
                    "Please either:\n"
                    "1. Add the files to your crystopo installation, or\n"
                    "2. Specify explicit paths in your config using 'skipatom_model_path' "
                    "and 'skipatom_training_data_path'"
                )
            config['skipatom_model_path'] = default_model_path
            config['skipatom_training_data_path'] = default_data_path

    return config

def initialize_descriptor(descriptor_type: str, analyzer: CrysToPoAnalyzer, config: Dict[str, Any]):
    """Initialize the requested descriptor type."""
    if descriptor_type == 'mace':
        model_path = config.get('mace_model_path', get_default_mace_path())
        if model_path is None:
            raise ValueError(
                "Could not find default MACE model file and no path provided in config.\n"
                "Please either:\n"
                "1. Add the file '2023-12-03-mace-128-L1_epoch-199.model' to your crystopo/data directory, or\n"
                "2. Specify explicit path in your config using 'mace_model_path'"
            )
        return MACECalculator(
            model_paths=model_path,
            device=config.get('device', 'cpu'),
            default_dtype="float32"
        )
    elif descriptor_type == 'chgnet':
        return CHGNet.load()
    elif descriptor_type == 'matminer':
        return SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
            stats=('mean', 'std_dev', 'minimum', 'maximum')
        )
    elif descriptor_type == 'soap':
        structures = get_all_structures(analyzer, config)
        species = get_unique_elements(structures)
        print(f"Initializing SOAP descriptor with species: {species}")
        return SOAP(
            species=species,
            periodic=True,
            r_cut=6.0,
            n_max=8,
            l_max=6,
            average="outer",
            compression={"mode": "mu2"}
        )
    elif descriptor_type == 'skipatom':
        print(f"Loading SkipAtom model from: {config['skipatom_model_path']}")
        print(f"Loading SkipAtom training data from: {config['skipatom_training_data_path']}")
        return SkipAtomInducedModel.load(
            config['skipatom_model_path'],
            config['skipatom_training_data_path'],
            min_count=2e7,
            top_n=5
        )
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")

def get_all_structures(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> List[Structure]:
    """Get all structures that will be used in the analysis."""
    all_structures = []
    method = config['method']

    if method == 'mp_query':
        if 'structures' in config:  # Structure classification
            for struct in config['structures']:
                structure_type = StructureType(
                    label=struct['label'],
                    formula_pattern=struct['formula_pattern'],
                    space_group=struct['space_group']
                )
                results = analyzer._fetch_and_process_structures(
                    structure_type.formula_pattern,
                    structure_type.space_group,
                    config.get('sample_size', None)
                )
                all_structures.extend([s for s, _ in results.values()])
        else:  # Metal or stability classification
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
            all_structures.extend([s for s, _ in results.values()])

    elif method == 'mp_ids':
        if 'structures' in config:  # Structure classification
            for mpids in config['structures'].values():
                for mpid in mpids:
                    structure, _ = analyzer._fetch_structure_and_density(mpid)
                    if structure is not None:
                        all_structures.append(structure)
        else:  # Metal or stability classification
            for mpid in config['mpids']:
                structure, _ = analyzer._fetch_structure_and_density(mpid)
                if structure is not None:
                    all_structures.append(structure)

    elif method == 'local_files':
        for label, group_data in config['data'].items():
            for struct_path in group_data['structures']:
                try:
                    structure = Structure.from_file(struct_path)
                    all_structures.append(structure)
                except Exception as e:
                    print(f"Failed to load structure from {struct_path}: {str(e)}")

    return all_structures

def get_unique_elements(structures: List[Structure]) -> List[str]:
    """Extract unique elements from all structures."""
    elements = set()
    for structure in structures:
        structure.remove_oxidation_states()
        elements.update([str(el) for el in structure.composition.elements])
    return list(elements)

def get_descriptors(structure, descriptor_type: str, descriptor):
    """Get descriptors for a structure using specified method."""
    if descriptor_type == 'mace':
        atoms = AseAtomsAdaptor().get_atoms(structure)
        features = descriptor.get_descriptors(
            atoms=atoms,
            invariants_only=True,
            num_layers=-1
        ).flatten()
        return (features, features, features)
    elif descriptor_type == 'chgnet':
        features = descriptor.predict_structure(
            structure,
            return_crystal_feas=True
        )['crystal_fea']
        return (features, features, features)
    elif descriptor_type == 'matminer':
        features = np.array(descriptor.featurize(structure))
        return (features, features, features)
    elif descriptor_type == 'soap':
        features = descriptor.create(structure.to_ase_atoms()).flatten()
        return (features, features, features)
    elif descriptor_type == 'skipatom':
        features = sum_pool(structure.composition, descriptor.dictionary, descriptor.vectors)
        return (features, features, features)

def run_structure_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run structure classification using chosen descriptor."""
    method = config['method']
    descriptor_type = config['descriptor_type']
    descriptor = initialize_descriptor(descriptor_type, analyzer, config)

    if method == 'mp_query':
        # Create StructureType objects and fetch structures
        structures_by_label = {}
        for struct in config['structures']:
            structure_type = StructureType(
                label=struct['label'],
                formula_pattern=struct['formula_pattern'],
                space_group=struct['space_group']
            )
            results = analyzer._fetch_and_process_structures(
                structure_type.formula_pattern,
                structure_type.space_group,
                config.get('sample_size', None)
            )
            structures_by_label[struct['label']] = [s for s, _ in results.values()]

    elif method == 'mp_ids':
        # Fetch structures by MPIDs
        structures_by_label = {}
        for label, mpids in config['structures'].items():
            structures = []
            for mpid in mpids:
                structure, _ = analyzer._fetch_structure_and_density(mpid)
                if structure is not None:
                    structures.append(structure)
            structures_by_label[label] = structures

    elif method == 'local_files':
        # Load structures from the nested data structure
        structures_by_label = {}
        for label, group_data in config['data'].items():
            structures = []
            for struct_path in group_data['structures']:
                try:
                    structure = Structure.from_file(struct_path)
                    structures.append(structure)
                    print(f"Loaded structure from {struct_path}")
                except Exception as e:
                    print(f"Failed to load structure from {struct_path}: {str(e)}")
            structures_by_label[label] = structures

    # Compute descriptors and prepare data for classifier
    data = {}
    for label, structures in structures_by_label.items():
        data[label] = []
        for structure in structures:
            try:
                features = get_descriptors(structure, descriptor_type, descriptor)
                data[label].append(features)
                print(f"Processed structure for {label}")
            except Exception as e:
                print(f"Failed to process structure: {str(e)}")

    # Train classifier
    classifier = BettiClassifier(n_estimators=100, random_state=42)
    train_acc, test_acc = classifier.train(data)

    # Create directory if it doesn't exist
    model_dir = os.path.dirname(config['model_path'])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    classifier.save(config['model_path'])

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_metal_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run metal classification using chosen descriptor."""
    method = config['method']
    descriptor_type = config['descriptor_type']
    descriptor = initialize_descriptor(descriptor_type, analyzer, config)

    structures = []
    if method in ['mp_query', 'mp_ids']:
        # Set the API key for the analyzer
        analyzer.config.api_key = config['mp_api_key']
        
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
            structures = [(mpid, s) for mpid, (s, _) in results.items()]
        else:  # mp_ids
            for mpid in config['mpids']:
                structure, _ = analyzer._fetch_structure_and_density(mpid)
                if structure is not None:
                    structures.append((mpid, structure))

    elif method == 'local_files':
        # Load structures from the nested data structure
        for label, group_data in config['data'].items():
            is_metal = group_data.get('is_metal', False)
            for struct_path in group_data['structures']:
                try:
                    structure = Structure.from_file(struct_path)
                    identifier = Path(struct_path).stem
                    structures.append((identifier, structure, is_metal))
                    print(f"Loaded structure from {struct_path}")
                except Exception as e:
                    print(f"Failed to load structure from {struct_path}: {str(e)}")

    # Compute descriptors and organize by metal/insulator
    data = {'metal': [], 'insulator': []}
    if method in ['mp_query', 'mp_ids']:
        for mpid, structure in structures:
            try:
                features = get_descriptors(structure, descriptor_type, descriptor)
                band_gap = mp_api.download_band_gap(mpid, config['mp_api_key'])
                category = 'metal' if band_gap == 0 else 'insulator'
                data[category].append(features)
                print(f"Processed {mpid} as {category}")
            except Exception as e:
                print(f"Failed to process {mpid}: {str(e)}")
    else:  # local_files
        for identifier, structure, is_metal in structures:
            try:
                features = get_descriptors(structure, descriptor_type, descriptor)
                category = 'metal' if is_metal else 'insulator'
                data[category].append(features)
                print(f"Processed {identifier} as {category}")
            except Exception as e:
                print(f"Failed to process {identifier}: {str(e)}")

    # Train classifier
    classifier = BettiClassifier(n_estimators=100, random_state=42)
    train_acc, test_acc = classifier.train(data)

    # Create directory if it doesn't exist
    model_dir = os.path.dirname(config['model_path'])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    classifier.save(config['model_path'])

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_stability_classification(analyzer: CrysToPoAnalyzer, config: Dict[str, Any]) -> None:
    """Run stability classification using chosen descriptor."""
    method = config['method']
    descriptor_type = config['descriptor_type']
    descriptor = initialize_descriptor(descriptor_type, analyzer, config)

    structures = []
    if method in ['mp_query', 'mp_ids']:
        # Set up MPRester
        if 'mp_api_key' not in config:
            raise ValueError("Materials Project API key required for MP methods")
        
        with MPRester(config['mp_api_key']) as mpr:
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
                for mpid, (structure, _) in results.items():
                    try:
                        data = mpr.get_entry_by_material_id(mpid)
                        e_above_hull = data.get('e_above_hull', None)
                        if e_above_hull is not None:
                            structures.append((mpid, structure, e_above_hull))
                    except Exception as e:
                        print(f"Error fetching stability data for {mpid}: {str(e)}")
                        continue
            else:  # mp_ids
                for mpid in config['mpids']:
                    try:
                        structure, _ = analyzer._fetch_structure_and_density(mpid)
                        data = mpr.get_entry_by_material_id(mpid)
                        e_above_hull = data.get('e_above_hull', None)
                        if structure is not None and e_above_hull is not None:
                            structures.append((mpid, structure, e_above_hull))
                    except Exception as e:
                        print(f"Error processing {mpid}: {str(e)}")
                        continue

    elif method == 'local_files':
        # Load structures from local files with provided stability labels
        if 'stability_labels' not in config['data']:
            raise ValueError("Stability labels must be provided for local files method")
        
        for struct_path, is_stable in zip(
            config['data']['structures'],
            config['data']['stability_labels']
        ):
            try:
                structure = Structure.from_file(struct_path)
                identifier = Path(struct_path).stem
                structures.append((identifier, structure, is_stable))
                print(f"Loaded structure from {struct_path}")
            except Exception as e:
                print(f"Failed to load structure from {struct_path}: {str(e)}")

    # Compute descriptors and organize by stability
    data = {'stable': [], 'unstable': []}
    for identifier, structure, stability_info in structures:
        try:
            features = get_descriptors(structure, descriptor_type, descriptor)
            if method in ['mp_query', 'mp_ids']:
                # For MP methods, stability_info is e_above_hull
                category = 'stable' if stability_info < 0.001 else 'unstable'
            else:
                # For local files, stability_info is a boolean
                category = 'stable' if stability_info else 'unstable'
            
            data[category].append(features)
            print(f"Processed {identifier} as {category}")
        except Exception as e:
            print(f"Failed to process {identifier}: {str(e)}")

    # Train classifier
    classifier = BettiClassifier(n_estimators=100, random_state=42)
    train_acc, test_acc = classifier.train(data)

    # Create directory if it doesn't exist
    model_dir = os.path.dirname(config['model_path'])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    classifier.save(config['model_path'])

    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Testing accuracy: {test_acc:.3f}")

def run_task(config: Dict[str, Any]) -> None:
    """Run the specified task with given configuration."""
    analyzer = CrysToPoAnalyzer()
    task = config['task']

    if task == 'structure_classification':
        run_structure_classification(analyzer, config)
    elif task == 'metal_classification':
        run_metal_classification(analyzer, config)
    elif task == 'stability_classification':
        run_stability_classification(analyzer, config)
    else:
        raise ValueError(f"Unknown task: {task}")

def main():
    """Run the benchmark script with the provided configuration."""
    if len(sys.argv) != 2:
        print("Usage: python benchmarks.py config.json")
        sys.exit(1)

    config = load_config(sys.argv[1])
    run_task(config)

if __name__ == '__main__':
    main()
