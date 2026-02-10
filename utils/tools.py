import rasterio 
import geopandas as gpd
import numpy as np

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


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
    """Reads a GeoJSON file into a GeoDataFrame using GeoPandas."""
    gdf = gpd.read_file(filepath_or_url)
    return gdf

def get_raster_data(file_path):
    # Open the file using a context manager
    with rasterio.open(file_path) as dataset:
        # Print some dataset properties
        print(f"Dataset name: {dataset.name}")
        print(f"File mode: {dataset.mode}")
        print(f"Number of bands: {dataset.count}")
        print(f"Image width: {dataset.width} pixels")
        print(f"Image height: {dataset.height} pixels")
        print(f"Coordinate Reference System (CRS): {dataset.crs}")

        cols, rows = np.meshgrid(np.arange(dataset.width), np.arange(dataset.height))
        xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)


        # Read all bands into a 3D NumPy array (bands, rows, columns)
        # If the image is single-band, the array will be 2D
        # The .read() method reads bands starting from index 1 (GDAL convention)
        image = dataset.read()

        print(f"Data shape: {image.shape}")
        print(f"Data type: {image.dtype}")

    return image, dataset, [lons, lats], [rows, cols]


def save_raster_and_write_meta(data, destination_tif, meta_source_tif):
        
    # Open the source GeoTIFF file in read mode ('r' is default)
    with rasterio.open(meta_source_tif) as src:
        # Read the raster data into a numpy array
        image_array = src.read() # Read the first band (adjust for multi-band rasters)

        # Get a copy of the source file's metadata (profile)
        profile = src.profile

        # Perform modifications on the numpy array
        # Example: Change all pixel values below 10000 to a new value (e.g., 9999)
        modified_array = data 

    # Update the profile for the output file
    # Ensure the dtype (data type) matches the modified array
    profile.update(
        dtype=modified_array.dtype,
        count=data.shape[0], # Number of bands (1 in this example)
        compress='lzw' # Optional: add compression
    )

    # Open the new GeoTIFF file in write mode ('w') and write the modified array
    with rasterio.open(destination_tif, 'w', **profile) as dst:
        dst.write(modified_array) # Write the modified array to band 1

    print(f"Modified image saved to {destination_tif}")