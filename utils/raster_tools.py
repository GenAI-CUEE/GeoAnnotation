import os
import cv2  
import rasterio   
import numpy as np 

from pyproj import Transformer  
from skimage import exposure

class Raster_profile:
    def __init__(self, mask_raster_file_path, center_geojson_file=None, center_geojson_crs="EPSG:4326"): 
        ''' 
        Usage: 
            - Read a complete info of a GeoTiff file using rasterio.

        Input: 
            - file_path: str. the filename of a GeoTif file that you wish to read. 

        Outputs:
            - self.raster: an np.array of tif image or mask
            - self.datainfo: data information 
        ''' 
            
        with rasterio.open(mask_raster_file_path) as datainfo:
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

        self.raster = image
        self.datainfo = datainfo 

    def convert_CHW_to_HWC(self, raster):
        return np.transpose(raster, (1, 2, 0)) 

    def get_profile(self):
        return self.datainfo.profile
    
    def get_longlat_from_image_pixels(self, y, x, crs_dst="EPSG:4326"):
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
    
    def image_enhancement(self, raster, max_pixel_value=255):
        normalized_img = cv2.normalize(raster, None, 0, max_pixel_value, cv2.NORM_MINMAX)  # Normalize to 0-1 range
        normalized_img_uint8 = normalized_img.astype(np.uint8)  

        red_channel = normalized_img_uint8[:, :, 0].ravel()
        green_channel = normalized_img_uint8[:, :, 1].ravel()
        blue_channel = normalized_img_uint8[:, :, 2].ravel()

        p2, p98 = np.percentile(red_channel, (2, 98))
        red_ch_rescale = exposure.rescale_intensity(red_channel, in_range=(p2, p98))

        p2, p98 = np.percentile(green_channel, (2, 98))
        green_ch_rescale = exposure.rescale_intensity(green_channel, in_range=(p2, p98))

        p2, p98 = np.percentile(blue_channel, (2, 98))
        blue_ch_rescale = exposure.rescale_intensity(blue_channel, in_range=(p2, p98))

        normalized_img_uint8[:,:,0] = red_ch_rescale.reshape(normalized_img_uint8.shape[0], normalized_img_uint8.shape[1])
        normalized_img_uint8[:,:,1] = green_ch_rescale.reshape(normalized_img_uint8.shape[0], normalized_img_uint8.shape[1])
        normalized_img_uint8[:,:,2] = blue_ch_rescale.reshape(normalized_img_uint8.shape[0], normalized_img_uint8.shape[1])

        return normalized_img_uint8