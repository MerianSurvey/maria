import os
import time
import glob
import socket
import numpy as np
import pandas as pd
from astropy import table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

# \\ set directory based on HOSTNAME
hostname = socket.gethostname()
if hostname == 'tigressdata2.princeton.edu':
    prefix = '/tiger/scratch/'
elif (hostname == 'tiger2-sumire.princeton.edu') | (hostname == 'tigercpu.princeton.edu'):
    prefix = '/scratch/'

_SUMMARY_CATALOG_PATHS = {1 : f"{prefix}gpfs/sd8758/merian/catalog/S20A/meriandr1_summary_catalog.fits"}
_TRACT_CATALOG_PATHS = {1 : f"{prefix}gpfs/sd8758/merian/catalog/S20A/$TRACTNUM/meriandr1_use_$TRACTNUM_S20A.fits"}

def read_summary_catalog ( dr=1, path=None, return_dataframe=True ):
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
    
    cat = table.Table(fits.getdata ( path, 1 ))
    
    if return_dataframe:
        return cat.to_pandas ().set_index('objectId_Merian')
    else:
        cat.add_index('objectId_Merian')
        return cat

def make_qualitymask ( catalog, code, **kwargs ):
    """
    Generate a quality mask for the catalog based on the provided code.

    Parameters:
        catalog (pandas.DataFrame): The catalog containing the data to be masked.
        code (str or array-like): Code used to create the mask.

    Returns:
        numpy.ndarray: A boolean mask representing the quality of the data.

    Note:
        - For 'use' code, the mask will be True for entries where 'SciUse' column equals 1.
        - For 'n708_detect' code, the mask will be True for entries where 'SciUse' column equals 1 and 
          'N708_gaap1p0Flux_Merian' is present, and the SNR of the N708 GaAP photometry is greater than 3.
        - For other codes, the mask will be created based on the provided code (TODO: more flexibility will be added in the future).
    """    
    if code == 'use':
        mask = catalog['SciUse'] == 1
    elif code == 'rmagcut':
        mask = catalog['SciUse'] == 1
        positive_rflux = catalog['r_cModelFlux_Merian']>0
        rflux = np.where ( positive_rflux,
                          catalog['r_cModelFlux_Merian'],
                          np.NaN)
        rmag = -2.5 * np.log10(rflux) + 31.4 # XXX IS THIS THE RIGHT ZP?
        if 'rmaglim' in kwargs.keys():
            rmaglim = kwargs['rmaglim']
        else:
            rmaglim = 23.
            
        mask &= rmag < rmaglim
    elif code == 'n708_detect':
        mask = catalog['SciUse'] == 1
        if 'N708_gaap1p0Flux_Merian' not in catalog.columns.names:
            mask = np.zeros_like(mask, dtype=bool)
        else:
            mask &= (catalog['N708_gaap1p0Flux_Merian']/catalog['N708_gaap1p0FluxErr_Merian']) > 3.
    elif code == "sdss_bright":
        # mask to get smaller catalog to match to SDSS
        mask = catalog["r_gaap1p0Flux_Merian"] > 3630 # in nJy --> rmag = 22.5
    elif code == "betsy_example":
        # select sources with good quality data 
        mask = catalog['SciUse'] == 1
        
        # select sources with good values in r
        positive_rflux = (catalog['r_cModelFlux_Merian']>0) & (catalog['r_cModelFluxErr_Merian']>0)
        rflux = np.where (positive_rflux,
                          catalog['r_cModelFlux_Merian'],
                          np.NaN)
        rfluxerr = np.where (positive_rflux,
                          catalog['r_cModelFluxErr_Merian'],
                          np.NaN)

        # calculate magnitude from flux using zero point
        rmag = -2.5 * np.log10(rflux) + 31.4

        # select sources brighter than a given magnitude value
        rmaglim = 23
        mask &= rmag < rmaglim

        # we can also do a SNR cut if we want
        SNR_r = rflux/rfluxerr
        SNRlim = 3
        mask &= SNR_r > SNRlim

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
    """
    Assemble a catalog by concatenating data from multiple tracts.

    Parameters:
        colnames (list): List of column names to be included in the assembled catalog.
        dr (int, optional): Data release number (default is 1).
        path (str, optional): Path to the tract catalog FITS file template. 
            If not provided, the path will be determined based on the data release number.
        usecode (str, optional): Code used to identify the tract catalog files (default is 'use').
        verbose (bool, optional): Whether to display verbose information (default is False).
        usescratch (bool, optional): Whether to use a scratch directory for caching assembled catalogs (default is True).
            * Note: the scratch filename is set by default as DR{dr}_{usecode}.csv TODO: expand flexibility
        scratchdir (str, optional): Directory path for storing cached catalogs (default is './scratch/').
    Returns:
        pandas.DataFrame: The assembled catalog containing data from all the specified tracts.
    Raises:
        KeyError: If the specified data release number is not recognized.
    """    
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
            return catalog
            
        
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
    
    #if usescratch:
    catalog.to_csv ( f'{scratchdir}/DR{dr}_{usecode}.csv')
    return catalog

def assemble_catalog_from_coordinates(coord_list, match_dist=.1*u.arcsec, 
                                      colnames=None, usecode = "use", verbose = False,
                                      dr=1, path=None, 
                                      usescratch=True,
                                      scratchdir='./scratch/'):  


    """
    Assemble a catalog by matching with a list of provided coordinates.

    Parameters:
        coord_list (2d list): List of coordinates to find merian matches for. Formatted (ra, dec)
        match_dist (float or astropy quantity): Upper limit for crossmatching distance. 
            If not an astropy quanitity, assumed to be in arcseconds
        colnames (list, optional): List of column names to be included in the assembled catalog. 
            If None, all columns will be included.
        dr (int, optional): Data release number (default is 1).
        path (str, optional): Path to the tract catalog FITS file template. 
            If not provided, the path will be determined based on the data release number.
        usecode (str, optional): Code used to identify the tract catalog files (default is 'use').
        verbose (bool, optional): Whether to display verbose information (default is False).
        usescratch (bool, optional): Whether to use a scratch directory for caching assembled catalogs (default is True).
            * Note: the scratch filename is set by default as DR{dr}_{usecode}_fromcoord.csv TODO: expand flexibility
        scratchdir (str, optional): Directory path for storing cached catalogs (default is './scratch/').
    Returns:
        pandas.DataFrame: The assembled catalog containing data from all the specified tracts.
        numpy.ndarray: An array of boolean values indicating whether the provided coordinate was matched to a Merian source.
        numpy.ndarray: An array of match distance to closest match for all provided coordinates. 
    Raises:
        KeyError: If the specified data release number is not recognized.
    """    

    # if no columns specified, use all columns
    if colnames is None:
        colnames = read_tract_catalog (9945, usecode=usecode, dr=dr, path=path).columns #9945 is small
        colnames = [x.name for x in colnames]

    if not type(match_dist) is u.quantity.Quantity:
        match_dist = match_dist * u.arcsec

    # read in the summary catalog to cross match
    sumcat = read_summary_catalog(dr=dr, path=path)

    # cross match
    if verbose:
        print ("Matching supplied coordinates with master catalog")

    ra, dec = coord_list
    _coords = SkyCoord(ra, dec, unit='deg')
    _sum = SkyCoord(sumcat['coord_ra_Merian'], sumcat['coord_dec_Merian'], unit='deg')
    ind_CtoM, dist_CtoM, _ = _coords.match_to_catalog_sky(_sum)

    # matched rows of summary catalog
    matched_sum = sumcat[ind_CtoM[dist_CtoM <= match_dist]]
    tracts = np.unique(matched_sum["tract"])
    # save a list of which coordinates didn't have merian matches
    coord_matched = dist_CtoM <= match_dist

    # make full catalog
    catalog = pd.DataFrame(index = matched_sum["objectId_Merian"], columns=colnames)
    if verbose:
        print (f"Compiling catalog for {len(matched_sum)} sources from {len(tracts)} tracts")
        start = time.time ()   
    for i, tract in enumerate(tracts):
        # read in tract with quality mask
        tract_cat = read_tract_catalog (tract, usecode=usecode, dr=dr, path=path)

        # get ids of matched sources in the tract
        tract_ids = matched_sum[matched_sum["tract"] == tract]['objectId_Merian']
        # some sources might have been filtered due ot the quality mask, so weed those out as well
        tract_ids = [id for id in tract_ids if id in tract_cat['objectId_Merian']]

        # find indices of those sources in the tract catalog
        tract_match_ind = [np.where(tract_cat['objectId_Merian'] == id)[0][0] for id in tract_ids]

        # save info to table
        for col in np.intersect1d(colnames, [x.name for x in tract_cat.columns]):
            catalog.loc[tract_ids, col] = tract_cat[tract_match_ind][col]

        if verbose and (i%50)==0:
            elapsed = time.time () - start
            print(f"Processed {i}/{len(tracts)} tracts after {elapsed:.2f} sec")

    if usescratch:
        catalog.to_csv ( f'{scratchdir}/DR{dr}_{usecode}_fromcoord.csv')

    return (catalog, coord_matched, dist_CtoM)
    
def assemble_catalog_conesearch(coord_center, seplimit=1*u.arcmin, 
                                colnames=None, usecode = "use", verbose = False,
                                dr=1, path=None, 
                                usescratch=True,
                                scratchdir='./scratch/'):  


    """
    Assemble a catalog by matching with a list of provided coordinates.

    Parameters:
        coord_center (list): Coordinates of cone center (ra, dec)
        seplimit (float or astropy.quantity): Upper limit on separation from center - i.e. cone radius.
            If not an astropy quanitity, assumed to be in arcseconds
        colnames (list, optional): List of column names to be included in the assembled catalog. 
            If None, all columns will be included.
        dr (int, optional): Data release number (default is 1).
        path (str, optional): Path to the tract catalog FITS file template. 
            If not provided, the path will be determined based on the data release number.
        usecode (str, optional): Code used to identify the tract catalog files (default is 'use').
        verbose (bool, optional): Whether to display verbose information (default is False).
        usescratch (bool, optional): Whether to use a scratch directory for caching assembled catalogs (default is True).
            * Note: the scratch filename is set by default as DR{dr}_{usecode}_fromcoord.csv TODO: expand flexibility
        scratchdir (str, optional): Directory path for storing cached catalogs (default is './scratch/').
    Returns:
        pandas.DataFrame: The assembled catalog containing data from within the specified cone. 
    Raises:
        KeyError: If the specified data release number is not recognized.
    """    

    # if no columns specified, use all columns
    if colnames is None:
        colnames = read_tract_catalog (9945, usecode=usecode, dr=dr, path=path).columns #9945 is small
        colnames = [x.name for x in colnames]

    if not type(seplimit) is u.quantity.Quantity:
        seplimit = seplimit * u.arcsec

    # read in the summary catalog to cross match
    sumcat = read_summary_catalog(dr=dr, path=path)

    # cross match
    if verbose:
        print ("Performing cone search")
    ra, dec = coord_center
    _coords = SkyCoord(ra, dec, unit='deg')
    _sum = SkyCoord(sumcat['coord_ra_Merian'], sumcat['coord_dec_Merian'], unit='deg')
    sep_dist = _sum.separation(_coords)
    close_ind = sep_dist < seplimit

    # matched rows of summary catalog
    matched_sum = sumcat[close_ind]
    tracts = np.unique(matched_sum["tract"])

    # make full catalog
    catalog = pd.DataFrame(index = matched_sum["objectId_Merian"], columns=colnames)
    if verbose:
        print (f"Compiling catalog for {len(matched_sum)} sources from {len(tracts)} tracts")
        start = time.time ()   
    for i, tract in enumerate(tracts):
        # read in tract with quality mask
        tract_cat = read_tract_catalog (tract, usecode=usecode, dr=dr, path=path)

        # get ids of matched sources in the tract
        tract_ids = matched_sum[matched_sum["tract"] == tract]['objectId_Merian']

        # find indices of those sources in the tract catalog
        tract_match_ind = [np.where(tract_cat['objectId_Merian'] == id)[0][0] for id in tract_ids]

        # save info to table
        for col in colnames:
            catalog.loc[tract_ids, col] = tract_cat[tract_match_ind][col]

        if verbose and (i%50)==0:
            elapsed = time.time () - start
            print(f"Processed {i}/{len(tracts)} tracts after {elapsed:.2f} sec")

    if usescratch:
        catalog.to_csv ( f'{scratchdir}/DR{dr}_{usecode}_conesearch.csv')

    return (catalog)
