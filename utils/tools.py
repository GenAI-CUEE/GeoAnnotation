import rasterio 
import geopandas as gpd
import numpy as np

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from pyproj import Transformer
from torchvision.ops import masks_to_boxes 
from shapely.geometry import box
from sklearn.metrics.pairwise import euclidean_distances

import cv2

ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        

def read_geojson(filepath_or_url):
    '''
    Usage: 
        Reads a GeoJSON file into a GeoDataFrame using GeoPandas.
    '''
    gdf = gpd.read_file(filepath_or_url)
    return gdf



def get_raster_data(file_path): 
    ''' 
    Usage: 
        - Read a complete info of a GeoTiff file using rasterio.

    Input: 
        - file_path: str. the filename of a GeoTif file that you wish to read. 

    Outputs:
        - image: an np.array of tif image  
        - datainfo: data information
        - [lons, lats]: latitude and longitude of the image array
        - [rows, cols]: row and column of the image array
    ''' 

    with rasterio.open(file_path) as datainfo:
        # Print some dataset properties
        print(f"Dataset name: {datainfo.name}")
        print(f"File mode: {datainfo.mode}")
        print(f"Number of bands: {datainfo.count}")
        print(f"Image width: {datainfo.width} pixels")
        print(f"Image height: {datainfo.height} pixels")
        print(f"Coordinate Reference System (CRS): {datainfo.crs.to_string()}")

        cols, rows = np.meshgrid(np.arange(datainfo.width), np.arange(datainfo.height))
        xs, ys = rasterio.transform.xy(datainfo.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)


        # Read all bands into a 3D NumPy array (bands, rows, columns)
        # If the image is single-band, the array will be 2D
        # The .read() method reads bands starting from index 1 (GDAL convention)
        image = datainfo.read()

        print(f"Data shape: {image.shape}")
        print(f"Data type: {image.dtype}")

    return image, datainfo


def get_raster_profile(meta_source_tif): 
    ''' 
    Usage: 
        Get the raster profile info.  

    Input:
        meta_source_tif: str. the filename of an existing file that you wish to know the geo profile information. 

    Output:
        - profile: dict. a dictionary contains the profile that is read from meta_source_tif.
    '''
    with rasterio.open(meta_source_tif) as src:
        # Read the raster data into a numpy array  
        # Get a copy of the source file's metadata (profile)
        profile = src.profile

    return profile


def save_raster_and_write_meta(data:np.array , destination_path: str, meta_source_tif: str=None, profile:dict=None):
    '''
    Usage: 
        Save a raster file in a GeoTiff format. 

    Inputs: 
        - data: np.array of raster images whose dimension is CxHxW
        - destination_path: str. the filename with .tif to be the destination path of the raster image to be saved.
        - meta_source_tif: str. the filename of an existing file that you wish to borrow the geo profile information. 
        If you don't have a meta_source_tif, you may alternatively specify the profile (a dictionary) as follows:
        
        - profile: a dictionary. An example is provided below ...
        ...   
            profile = {
                'driver': 'GTiff',
                'height': 100,
                'width': 100,
                'count': 1,
                'dtype': 'uint8',
                'crs': 'EPSG:3857', 
                'compress': 'lzw' # Optional compression
            }
    
    Output: None. A GeoTiff file will be saved at the destination path (destination_path). 
    '''


    if meta_source_tif is None and profile:
        print("meta_source_tif: the filename of an existing file that share similar geo information. \n")
        print("or a profile (dictionary) is needed...")  

        ValueError


    if meta_source_tif is not None:   
        # Open the source GeoTIFF file in read mode ('r' is default)
        with rasterio.open(meta_source_tif) as src:
            # Read the raster data into a numpy array  
            # Get a copy of the source file's metadata (profile)
            profile = src.profile

        # Perform modifications on the numpy array
        # Example: Change all pixel values below 10000 to a new value (e.g., 9999)

    # Update the profile for the output file
    # Ensure the dtype (data type) matches the modified array
    profile.update(
        dtype=data.dtype,
        count=data.shape[0], # Number of bands (1 in this example) 
        compress='lzw' # Optional: add compression
    )

    # Open the new GeoTIFF file in write mode ('w') and write the modified array
    with rasterio.open(destination_path, 'w', **profile) as dst:
        dst.write(data) # Write the modified array to band 1

    print(f"Modified image saved to : {destination_path}")

 