import numpy as np
import pandas as pd
from astropy import table
from astropy.io import fits

_SUMMARY_CATALOG_PATHS = {1 : "/scratch/gpfs/sd8758/merian/catalog/S20A/meriandr1_master_catalog.fits"}
_TRACT_CATALOG_PATHS = {1 : "/scratch/gpfs/sd8758/merian/catalog/S20A/$TRACTNUM/meriandr1_use_$TRACTNUM_S20A.fits"}

def read_summary_catalog ( dr=1, path=None ):
    """
    Read the summary catalog data for a specified data release.

    Parameters:
        dr (int, optional): Data release number (default is 1).
        path (str, optional): Path to the summary catalog FITS file. 
        If not provided, the path will be determined based on the data release number.

    Returns:
        astropy.io.fits.fitsrec.FITS_rec: The summary catalog data loaded from the FITS file.

    Raises:
        KeyError: If the specified data release number is not recognized.
    """    
    if path is  None:        
        if dr not in _SUMMARY_CATALOG_PATHS.keys():
            raise KeyError (f"Data release {dr} not recognized!")
        path = _SUMMARY_CATALOG_PATHS[dr]
    
    cat = fits.getdata ( path, 1 )
    return cat
        
def read_tract_catalog (tractnumber, dr=1, path=None):
    """
    Read the tract catalog data for a specified tract number and data release.

    Parameters:
        tractnumber (int): The tract number for the catalog.
        dr (int, optional): Data release number (default is 1).
        path (str, optional): Path to the tract catalog FITS file template. 
        If not provided, the path will be determined based on the data release number.

    Returns:
        numpy.ndarray: The tract catalog data loaded from the FITS file.

    Raises:
        KeyError: If the specified data release number is not recognized.
    """    
    if path is  None:        
        if dr not in _TRACT_CATALOG_PATHS.keys():
            raise KeyError (f"Data release {dr} not recognized!")
        path = _TRACT_CATALOG_PATHS[dr]    
        
    cat = fits.getdata (path.replace("$TRACTNUM",str(tractnumber)), 1)
    return cat

