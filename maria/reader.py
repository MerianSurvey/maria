import os
import time
import glob
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

def make_qualitymask ( catalog, code ):
    if code == 'use':
        mask = catalog['SciUse'] == 1
    elif code == 'n708_detect':
        mask = catalog['SciUse'] == 1
        if 'N708_gaap1p0Flux_Merian' not in catalog.columns.names:
            mask = np.zeros_like(mask, dtype=bool)
        else:
            mask &= (catalog['N708_gaap1p0Flux_Merian']/catalog['N708_gaap1p0FluxErr_Merian']) > 3.
    else:
        mask = code # TODO: allow for more flexibility, need to come back to this
    return mask
        
def read_tract_catalog (tractnumber, dr=1, path=None, usecode='use'):
    """
    Read the tract catalog data for a specified tract number and data release.

    Parameters:
        tractnumber (int/str): The tract number for the catalog.
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
    
    # \\ apply science use-case mask
    usemask = make_qualitymask ( cat, usecode )
    cat = cat[usemask]
    return cat

def assemble_catalog ( colnames, dr=1, path=None, usecode='use', verbose=False,
                       usescratch=True,
                       scratchdir='./scratch/'):
    if path is  None:        
        if dr not in _TRACT_CATALOG_PATHS.keys():
            raise KeyError (f"Data release {dr} not recognized!")
        path = _TRACT_CATALOG_PATHS[dr]
    if usescratch:
        scratchfile = f'{scratchdir}/DR{dr}_{usecode}.csv'
        if os.path.exists (scratchfile):
            catalog = pd.read_csv ( scratchfile, index_col=0 )
            if verbose:
                print(f'Warning: reading from scratch {scratchfile}')
            
        
    available_tracts = [ os.path.basename(x[:-1]) for x in glob.glob(path.split("$TRACTNUM")[0] + '/*/') ]
    # \\ need to cut out chafe directories like old/
    available_tracts = [ x for x in available_tracts if x.isdigit ()]
    
    if verbose:
        print(f"Concatenating {len(available_tracts)} tracts")
    
    dataframes = [] #pd.DataFrame ( columns=colnames)
    i=0
    if verbose:
        start = time.time ()
    for tract in available_tracts:
        tract_catalog = read_tract_catalog (tract, usecode=usecode)
        tract_columns = [ x.name for x in tract_catalog.columns ]

        ids = tract_catalog['objectId_Merian']
        df = pd.DataFrame ( index = ids, columns=colnames)
        for col in colnames:
            if col in tract_columns:
                df.loc[ids, col] = tract_catalog[col]
            else:
                pass
                #if verbose:
                #    print ( f"{col} not in tract catalog {tract}!")
        dataframes.append(df)
        i+=1
        if verbose and (i%50)==0:
            elapsed = time.time () - start
            print(f"Processed {i}/{len(available_tracts)} tracts after {elapsed:.2f} sec")
    catalog = pd.concat(dataframes)
    
    if usescratch:
        catalog.to_csv ( f'{scratchdir}/DR{dr}_{usecode}.csv')
    return catalog

