import sys
import rasterio as rio
import numpy as np
from pyproj.crs import CRS
from scipy import ndimage
from tqdm import tqdm
from collections import Counter


#%%% FUNCTIONS
def checkFormat(file, listExtensions=['nc','tif']):
    if not (file.split(".")[-1] in listExtensions):
        raise Exception(" (!) Input file {0} must be a *.tif or *.nc".format(file))

def createWseMask(hmax_path, field_path='', outfile_path='', threshold=0.1, min_patch_size=10, nodataVal=-9999, write=True, write_mask=False):
    '''
        BG-Flood output file mask function. Creates a mask to remove isolated rainfall patches, and applies it to the specified raster/map.
            1. Initialise the mask as hmax > hmax_threshold
            2. Excludes all wet patches composed of less than min_path_size cells
            3. Applies the mask to field_path and writes the output
    '''
    ## Check if hmax_path is nc or tif
    checkFormat(hmax_path)
    
    if not outfile_path:
        outfile_path = hmax_path[:-(len(hmax_path.split(".")[-1])+1)] + '_masked.tif'
    
    outfile_mask_path = outfile_path[:-(len(outfile_path.split("/")[-1])+1)] + '/mask_' + outfile_path.split("/")[-1]
    
    ## Load hmaxfile
    with rio.open(hmax_path, 'r') as ds_hmax:
        hmax_raw = ds_hmax.read(1, masked=False)
        hmax_raw = np.nan_to_num(hmax_raw)
        profile = ds_hmax.profile
        
        ### Remove nans
    hmax_raw = np.nan_to_num(hmax_raw)
      
    hmax = (hmax_raw > threshold)*hmax_raw
    
    ## Structure of the links connecting pixels -- Here neighbours and diagonals are considered part of the same feature
    struct = np.ones((3,3)) 
    labels, num_features = ndimage.label(hmax, structure=struct)

    ## Filter isolated patches
    mask = np.ones_like(hmax)
    labels_counter = Counter(list(labels.flatten()))
    
    for label, count in tqdm(labels_counter.items(), total=len(range(1, num_features + 1))):
        if count >= min_patch_size:
            mask[labels == label] = 0

        ### Load and apply filter to var to filter
    if field_path == "":
        var_raw = hmax_raw.copy()
    else:
        with rio.open(field_path, 'r') as ds_field:
            var_raw = ds_field.read(1, masked=False)
            var_raw = np.nan_to_num(var_raw)
            profile = ds_field.profile
    
    #var = var_raw*(hmax_raw > threshold)*(1-mask)
    var = var_raw*(1-mask)
    
            ### Prepare and write files
    if write:
        profile.update(
            driver='GTiff',
            crs=CRS.from_epsg(2193),
            nodata=nodataVal,
            compress='lzw',
            #dtype=rio.float32,
            )

        with rio.open(outfile_path, 'w', **profile) as dsout:
            dsout.write(var, 1)
            dsout.nodata = nodataVal
        
        if write_mask:
            with rio.open(outfile_mask_path, 'w', **profile) as dsout:
                dsout.write(mask, 1)
                dsout.nodata = nodataVal
            
    return var
    
    
    
#%%% MAIN    

hmax_path    = sys.argv[1]  # Water surface level map (hmax)
field_path   = sys.argv[2]  # Map to filter
outfile_path = sys.argv[3]  # Output file

#workdir = "E:/07_Flood_modelling_NIWA/02_Simulations/Napier/run_26_highRoughness_Redclyffe/new/"

#hmax_path = workdir + "hmax_259200s_Napier_06_new.nc"
#field_path = hmax_path
#outfile_path = workdir + "masked_Heretaunga_Plains_100y_48h_0c_Inundation_Floodmap_hmax_nztm_p200.tif"

    
hmax_threshold = 0.05 # in m
# Maximum number of cells in an object to be filtered - To calibrate depending on input map resolution and hmax_threshold
#   Good initial values I usually use:
#       - 32m resolution: min_patch_size ~ 5 - 10
#       - 16m resolution: min_patch_size ~ 15 - 25 
#       - 8m resolution:  min_patch_size ~ 50 - 100 (conservative, can go higher)
#       - 4m resolution:  min_patch_size = 100 - 150 (conservative, can go higher)
min_patch_size = 50
nodataVal = -9999

if __name__ == "__main__":    
    createWseMask(
        hmax_path,
        field_path=field_path,
        outfile_path=outfile_path,
        threshold=hmax_threshold,
        min_patch_size=min_patch_size,
        nodataVal=nodataVal
    )
    
    

