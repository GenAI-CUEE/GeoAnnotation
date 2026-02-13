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



class Mask_Raster:
    def __init__(self, file_path): 
        ''' 
        Usage: 
            - Read a complete info of a GeoTiff file using rasterio.

        Input: 
            - file_path: str. the filename of a GeoTif file that you wish to read. 

        Outputs:
            - self.mask: an np.array of tif image or mask
            - self.datainfo: data information 
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

        self.mask = image
        self.datainfo = datainfo
        
    def get_profile(self):
        return self.datainfo.profile
    
    def convert_pixel_to_longlat(self, y, x, crs_dst="EPSG:4326"):
        long, lat = self.datainfo.xy(y, x)

        if not(self.datainfo.crs.to_string() == crs_dst): 
            transformer = Transformer.from_crs(self.datainfo.crs, crs_dst, always_xy=True)
            long, lat = transformer.transform(long, lat)

        return long, lat    
    
    def get_TL_BR_corners_in_latlong(self, crs_dst="EPSG:4326"):  

        '''
            TL (top left) / BR (bottom right) from tif_data data.
            Inputs:
                tif_data: obj from rasterio.open.
                crs_dst: target crs system.
            Outputs:
                top_left_longlat: a list of long, lat corresponding to the top left corner. 
                bottom_right_longlat: a list of long, lat corresponding to the bottom right corner.
        '''
        lon_start, lat_start = self.datainfo.xy(0, 0)
        lon_end, lat_end = self.datainfo.xy(self.datainfo.height, self.datainfo.width)
            
        if not(self.datainfo.crs.to_string() == crs_dst):  
            
            transformer = Transformer.from_crs(self.datainfo.crs, crs_dst, always_xy=True)
            lon_start, lat_start = transformer.transform(lon_start, lat_start)

            transformer = Transformer.from_crs(self.datainfo.crs, crs_dst, always_xy=True)
            lon_end, lat_end = transformer.transform(lon_end, lat_end)

        top_left_longlat     = [lon_start, lat_start] # long, lat 
        bottom_right_longlat = [lon_end, lat_end]  # long, lat  
                
        return top_left_longlat, bottom_right_longlat 


    def get_image_pixels_from_longlat(self, long, lat, crs_dst="EPSG:4326"):  

        top_left_longlat, bottom_right_longlat = self.get_TL_BR_corners_in_latlong(crs_dst)

        image_height = self.datainfo.height
        image_width  = self.datainfo.width

        
        min_long = min(top_left_longlat[0], bottom_right_longlat[0])
        max_long = max(top_left_longlat[0], bottom_right_longlat[0])

        min_lat  = min(top_left_longlat[1], bottom_right_longlat[1])
        max_lat  = max(top_left_longlat[1], bottom_right_longlat[1]) 

        pixel_x =  ((long - min_long) / (max_long - min_long))*image_width  
        pixel_y =  (1-(lat - min_lat) / (max_lat - min_lat))*image_height  
        pixel = [int(pixel_x), int(pixel_y)] 

        return pixel
    

    def make_boundboxes(self, center_geojson, crs_dst="EPSG:4326"):
 
        mask_uint8 = self.mask.astype(np.uint8)

        center   = read_geojson(center_geojson)
        center["longitude"] = center.geometry.x
        center["latitude"]  = center.geometry.y

        pixel_x = []
        pixel_y = []
        for long, lat in zip(center["longitude"].tolist(), center["latitude"].tolist()):
            pixel_xy = self.get_image_pixels_from_longlat(long, lat, crs_dst=crs_dst)
            pixel_x.append(pixel_xy[0])
            pixel_y.append(pixel_xy[1])

        
        tensor = torch.tensor(mask_uint8[0,:,:], dtype=torch.uint8)
        label_id = torch.unique(tensor)
        torch_masks_list = []
        for id in label_id.tolist(): 
            bool_image = (1*(tensor == id)) 
            torch_masks_list.append(bool_image.view(1, tensor.shape[0], tensor.shape[1]))

        torch_masks = torch.concat(torch_masks_list)
        boxes = masks_to_boxes(torch_masks)

        boxes_np =  boxes[1:,:].numpy().tolist()

        # plt.imshow(mask_uint8[0,:,:], cmap='turbo')
        # plt.scatter(pixel_x, pixel_y, c='cyan', s=10)
        # Iterate through boxes and add patches 

        boxes_np_list = list(boxes_np)

        longlat_boxes = []

        for x_min, y_min, x_max, y_max in boxes_np_list:
            lon_min, lat_min = self.convert_pixel_to_longlat(y_min, x_min, crs_dst=crs_dst)
            lon_max, lat_max = self.convert_pixel_to_longlat(y_max, x_max, crs_dst=crs_dst)

            longlat_boxes.append([lon_min, lat_min, lon_max, lat_max]) 

        polygons = [box(minx, miny, maxx, maxy) for minx, miny, maxx, maxy in longlat_boxes]

        # 3. Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
        
