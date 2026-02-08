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


def copy_meta_and_write_raster(source_tif, destination_tif):
    with rasterio.open(source_tif) as src:
        # 2. Get the metadata (including CRS, transform, dimensions, etc.)
        # and create a copy of it
        dst_kwargs = src.meta.copy()
        dst_kwargs['dtype'] = 'uint8'  # Update data type if necessary

        # If you are writing a different NumPy array (e.g., of different data type or band count),
        # you may need to update 'dtype' and 'count' in dst_kwargs
        # dst_kwargs.update({'dtype': data_array.dtype, 'count': 1})

        # 3. Open the destination file in write mode
        with rasterio.open(destination_tif, 'w', **dst_kwargs) as dst:
            # 4. Write your data to the destination file
            # This example writes the data from the source file (band 1)
            # If using a new array, it would be: dst.write(data_array, 1)
            dst.write(src.read(1), 1)