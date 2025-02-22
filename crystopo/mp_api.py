from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import numpy.typing as npt
from pymatgen.core import Structure
from mp_api.client import MPRester
from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
import multiprocessing
import os

class MPInterface:
    """Interface for downloading data from the Materials Project."""
    
    load_dotenv()
    def __init__(self, api_key: str = os.getenv('MP_API_KEY')) -> None:
        """
        Initialize the API interface.
        
        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
    
    def _get_charge_density_base(self, mpid: str) -> Optional[npt.NDArray[np.float64]]:
        """Base function to get charge density using MPRester."""
        try:
            with MPRester(self.api_key) as mpr:
                chgcar = mpr.get_charge_density_from_material_id(mpid)
                if chgcar is None:
                    return None
                chg_dict = chgcar.as_dict()
                return np.array(chg_dict['data']['total'])
        except Exception as e:
            print(f"Error retrieving data for {mpid}: {str(e)}")
            return None
    
    def _worker(self, mpid: str) -> Optional[npt.NDArray[np.float64]]:
        """Worker function to run in a separate process."""
        pool = ThreadPool(1)  # Use thread pool to encapsulate network call
        future = pool.apply_async(self._get_charge_density_base, (mpid,))
        try:
            return future.get(timeout=20)  # timeout in seconds
        except multiprocessing.TimeoutError:
            print(f"Timeout occurred while retrieving data for {mpid}")
            return None
        finally:
            pool.close()
            pool.join()
    
    def get_charge_density(self,
                          mpid: str) -> Optional[npt.NDArray[np.float64]]:
        """
        Download charge density for a single material.
        
        Args:
            mpid: Materials Project ID (e.g., 'mp-149')
            
        Returns:
            3D numpy array of charge density values, or None if download fails
        """
        try:
            with multiprocessing.Pool(1) as proc_pool:
                result = proc_pool.apply_async(self._worker, (mpid,))
                chgcar = result.get(timeout=30)  # slightly longer than worker timeout
                
                if chgcar is None:
                    print(f"No charge density data available for {mpid}")
                    return None
                    
                return chgcar
                
        except multiprocessing.TimeoutError:
            print(f"Download for {mpid} timed out. This usually indicates "
                  "the API is stuck retrieving too many documents.")
            return None
        except Exception as e:
            print(f"Error downloading charge density for {mpid}: {str(e)}")
            return None
    
    def get_charge_densities(self,
                            mpids: List[str]
                            ) -> Dict[str, Optional[npt.NDArray[np.float64]]]:
        """
        Download charge densities for multiple materials.
        Continues with remaining materials if one fails.
        
        Args:
            mpids: List of Materials Project IDs
            
        Returns:
            Dictionary mapping MPIDs to their charge density arrays
            (None for materials without charge density data)
        """
        results = {}
        unavailable = []
        
        for mpid in mpids:
            print(f"Attempting to download charge density for {mpid}...")
            results[mpid] = self.get_charge_density(mpid)
            if results[mpid] is None:
                unavailable.append(mpid)
                print(f"Moving on to next material...")
        
        # Summary at the end
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\nDownload summary:")
        print(f"Successfully downloaded: {successful}/{len(mpids)}")
        if unavailable:
            print(f"Materials without charge density data: {', '.join(unavailable)}")
        
        return results

    def find_mpids(self,
                   composition_pattern: str,
                   spacegroup: int,
                   ehull_max: float = 1.0) -> List[str]:
        """
        Find MPIDs matching a composition pattern and space group.

        Args:
            composition_pattern: String pattern like "*1*2" for compositions
                               like Ti1O2, Ru1O2, etc.
            spacegroup: Space group number
            ehull_max: Maximum energy above hull in eV/atom

        Returns:
            List of matching MPIDs
        """
        try:
            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(
                    formula=composition_pattern,
                    energy_above_hull=(None, ehull_max),
                    spacegroup_number=spacegroup,
                    fields=['material_id']
                )
                return [entry.material_id for entry in docs]
        except Exception as e:
            print(f"Error searching Materials Project: {str(e)}")
            return []

    def get_structure(self, mpid: str) -> Optional[Structure]:
        """
        Download structure for a single material.

        Args:
            mpid: Materials Project ID (e.g., 'mp-149')

        Returns:
            Pymatgen Structure object, or None if download fails
        """
        try:
            with MPRester(self.api_key) as mpr:
                structure = mpr.get_structure_by_material_id(mpid)
                return structure
        except Exception as e:
            print(f"Error downloading structure for {mpid}: {str(e)}")
            return None

    def get_structures(self,
                      mpids: List[str]
                      ) -> Dict[str, Optional[Structure]]:
        """
        Download structures for multiple materials.

        Args:
            mpids: List of Materials Project IDs

        Returns:
            Dictionary mapping MPIDs to their Structure objects
            (None for failed downloads)
        """
        results = {}
        for mpid in mpids:
            print(f"Downloading structure for {mpid}...")
            results[mpid] = self.get_structure(mpid)
        return results

    def get_structure_and_charge_density(self,
                                       mpid: str
                                       ) -> Tuple[Optional[Structure],
                                                Optional[npt.NDArray[np.float64]]]:
        """
        Download both structure and charge density for a single material.

        Args:
            mpid: Materials Project ID

        Returns:
            Tuple of (Structure, charge_density_array), either can be None if download fails
        """
        structure = self.get_structure(mpid)
        charge_density = self.get_charge_density(mpid)
        return structure, charge_density

    def get_structures_and_charge_densities(self,
                                          mpids: List[str]
                                          ) -> Dict[str,
                                                  Tuple[Optional[Structure],
                                                       Optional[npt.NDArray[np.float64]]]]:
        """
        Download both structures and charge densities for multiple materials.

        Args:
            mpids: List of Materials Project IDs

        Returns:
            Dictionary mapping MPIDs to tuples of (Structure, charge_density_array)
        """
        results = {}
        for mpid in mpids:
            print(f"Downloading data for {mpid}...")
            results[mpid] = self.get_structure_and_charge_density(mpid)
        return results

    def get_band_gap(self, mpid: str) -> Optional[float]:
        """
        Download band gap for a single material.

        Args:
            mpid: Materials Project ID (e.g., 'mp-149')

        Returns:
            Band gap in eV, or None if download fails
        """
        try:
            with MPRester(self.api_key) as mpr:
                data = mpr.summary.get_data_by_id(mpid, fields=['band_gap'])
                if data:
                    return data.band_gap
                print(f"No band gap data available for {mpid}")
                return None
        except Exception as e:
            print(f"Error downloading band gap for {mpid}: {str(e)}")
            return None

    def get_band_gaps(self, mpids: List[str]) -> Dict[str, Optional[float]]:
        """
        Download band gaps for multiple materials.
        Continues with remaining materials if one fails.

        Args:
            mpids: List of Materials Project IDs

        Returns:
            Dictionary mapping MPIDs to their band gaps in eV
            (None for materials without band gap data)
        """
        results = {}
        unavailable = []

        for mpid in mpids:
            print(f"Downloading band gap for {mpid}...")
            results[mpid] = self.get_band_gap(mpid)
            if results[mpid] is None:
                unavailable.append(mpid)
                print(f"Moving on to next material...")

        # Summary at the end
        successful = sum(1 for v in results.values() if v is not None)
        print(f"\nDownload summary:")
        print(f"Successfully downloaded: {successful}/{len(mpids)}")
        if unavailable:
            print(f"Materials without band gap data: {', '.join(unavailable)}")

        return results

def find_materials(composition_pattern: str,
                  spacegroup: int,
                  api_key: str,
                  ehull_max: float = 1.0) -> List[str]:
    """
    Convenience function to find MPIDs by composition and space group.

    Args:
        composition_pattern: String pattern like "*1*2" for compositions
                           like Ti1O2, Ru1O2, etc.
        spacegroup: Space group number
        api_key: Materials Project API key
        ehull_max: Maximum energy above hull in eV/atom

    Returns:
        List of matching MPIDs
    """
    api = MPInterface(api_key)
    return api.find_mpids(composition_pattern, spacegroup, ehull_max)

def download_structure(mpid: Union[str, List[str]],
                      api_key: str
                      ) -> Union[Optional[Structure],
                               Dict[str, Optional[Structure]]]:
    """
    Convenience function to download structure data.

    Args:
        mpid: Single MPID or list of MPIDs
        api_key: Materials Project API key

    Returns:
        Single Structure object or dictionary of Structure objects
    """
    api = MPInterface(api_key)

    if isinstance(mpid, str):
        return api.get_structure(mpid)
    else:
        return api.get_structures(mpid)

def download_charge_density(mpid: Union[str, List[str]],
                          api_key: str
                          ) -> Union[Optional[npt.NDArray[np.float64]],
                                   Dict[str, Optional[npt.NDArray[np.float64]]]]:
    """
    Convenience function to download charge density data.

    Args:
        mpid: Single MPID or list of MPIDs
        api_key: Materials Project API key

    Returns:
        Single charge density array or dictionary of arrays
    """
    api = MPInterface(api_key)

    if isinstance(mpid, str):
        return api.get_charge_density(mpid)
    else:
        return api.get_charge_densities(mpid)

def download_structure_and_charge_density(
    mpid: Union[str, List[str]],
    api_key: str
) -> Union[Tuple[Optional[Structure], Optional[npt.NDArray[np.float64]]],
           Dict[str, Tuple[Optional[Structure], Optional[npt.NDArray[np.float64]]]]]:
    """
    Convenience function to download both structure and charge density data.

    Args:
        mpid: Single MPID or list of MPIDs
        api_key: Materials Project API key

    Returns:
        For single MPID: Tuple of (Structure, charge_density_array)
        For list of MPIDs: Dictionary mapping MPIDs to such tuples
    """
    api = MPInterface(api_key)

    if isinstance(mpid, str):
        return api.get_structure_and_charge_density(mpid)
    else:
        return api.get_structures_and_charge_densities(mpid)

def download_band_gap(mpid: Union[str, List[str]],
                     api_key: str
                     ) -> Union[Optional[float],
                              Dict[str, Optional[float]]]:
    """
    Convenience function to download band gap data.

    Args:
        mpid: Single MPID or list of MPIDs
        api_key: Materials Project API key

    Returns:
        Single band gap value in eV or dictionary of band gaps
    """
    api = MPInterface(api_key)

    if isinstance(mpid, str):
        return api.get_band_gap(mpid)
    else:
        return api.get_band_gaps(mpid)
