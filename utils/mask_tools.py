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
from utils.tools import read_geojson
import cv2 


def filling_holes(gray):

    des = cv2.bitwise_not(gray)
    contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des,[cnt],0, 255,-1)

    gray = cv2.bitwise_not(des)

    return gray

def erosion(gray, kernel_size=5):

    # Creating kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Using cv2.erode() method 
    gray = cv2.erode(gray, kernel, iterations=1) 

    return gray

def dilation(gray, kernel_size=5):

    # Creating kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Using cv2.erode() method 
    gray = cv2.dilate(gray, kernel) 

    return gray
 

class Mask_profile:
    def __init__(self, mask_raster_file_path, center_geojson_file=None, center_geojson_crs="EPSG:4326"): 
        ''' 
        Usage: 
            - Read a complete info of a GeoTiff file using rasterio.

        Input: 
            - file_path: str. the filename of a GeoTif file that you wish to read. 

        Outputs:
            - self.mask: an np.array of tif image or mask
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

        self.mask = image[0, :, :]
        self.datainfo = datainfo

        self.center_geojson_file = None 
        self.center_geojson_crs = None
        self.center_pixel_x = None
        self.center_pixel_y = None
        self.center_longitude = None 
        self.center_latitude  = None

        if center_geojson_file is not None:
            center   = read_geojson(center_geojson_file) 
            self.center_geojson_file = center_geojson_file
            self.center_geojson_crs = center_geojson_crs
            self.center_longitude = center.geometry.x
            self.center_latitude  = center.geometry.y 

            pixel_x = []
            pixel_y = []
            for long, lat in zip(self.center_longitude.tolist(), self.center_latitude.tolist()):
                pixel_xy = self.get_image_pixels_from_longlat(long, lat, crs_dst=center_geojson_crs)
                pixel_x.append(pixel_xy[0])
                pixel_y.append(pixel_xy[1])

            self.center_pixel_x = pixel_x
            self.center_pixel_y = pixel_y
        
        self.record_list = [] # [{"mask_id": None,"Operation": None, "Kernel": None}]

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
    

    def make_boundboxes(self, crs_dst="EPSG:4326"):
        mask_uint8 = self.mask.astype(np.uint8) 

        tensor = torch.tensor(mask_uint8, dtype=torch.uint8)
        label_id = torch.unique(tensor)
        torch_masks_list = []
        for id in label_id.tolist(): 
            bool_image = (1*(tensor == id)) 
            torch_masks_list.append(bool_image.view(1, tensor.shape[0], tensor.shape[1]))

        torch_masks = torch.concat(torch_masks_list)
        boxes = masks_to_boxes(torch_masks)

        boxes_np =  boxes[1:,:].numpy().tolist()

        # plt.imshow(mask_uint8, cmap='turbo')
        # plt.scatter(pixel_x, pixel_y, c='cyan', s=10)
        # Iterate through boxes and add patches 

        boxes_np_list = list(boxes_np)

        longlat_boxes = []

        for x_min, y_min, x_max, y_max in boxes_np_list:
            lon_min, lat_min = self.get_longlat_from_image_pixels(y_min, x_min, crs_dst=crs_dst)
            lon_max, lat_max = self.get_longlat_from_image_pixels(y_max, x_max, crs_dst=crs_dst)

            longlat_boxes.append([lon_min, lat_min, lon_max, lat_max]) 

        polygons = [box(minx, miny, maxx, maxy) for minx, miny, maxx, maxy in longlat_boxes]

        # 3. Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs_dst) 
        # gdf.to_file(destination_bbx_filename.geojson, driver='GeoJSON') 
        return gdf


    def mapping_geojson_center_to_mask_id(self, center_geojson_file=None, center_geojson_crs="EPSG:4326"): 

        if center_geojson_file is not None:
            center   = read_geojson(center_geojson_file) 
            self.center_geojson_file = center_geojson_file
            self.center["longitude"] = center.geometry.x
            self.center["latitude"]  = center.geometry.y  

            pixel_x = []
            pixel_y = []
            for long, lat in zip(self.center["longitude"].tolist(), self.center["latitude"].tolist()):
                pixel_xy = self.get_image_pixels_from_longlat(long, lat, crs_dst=center_geojson_crs)
                pixel_x.append(pixel_xy[0])
                pixel_y.append(pixel_xy[1])

            self.center_pixel_x = pixel_x
            self.center_pixel_y = pixel_y
        

        self.mapping_gjcenter_id_to_mask_id = {}
        self.mapping_mask_id_to_gjcenter_id = {}

        gjcenter_index_list = []
        for mask_index in range(1, self.mask.max()+1): 
            gray_bf                = 1*(self.mask == mask_index)  # Convert to grayscale by taking one channel
            (gray_bf_y, gray_bf_x) = np.nonzero(gray_bf)
            gray_bf_xy             = np.concatenate([gray_bf_x.reshape(-1,1), gray_bf_y.reshape(-1,1)], axis=1)  
            ref_xy                 = np.concatenate([np.array(self.center_pixel_x).reshape(-1,1), np.array(self.center_pixel_y).reshape(-1,1)], axis=1) 
            distances              = euclidean_distances(gray_bf_xy, ref_xy)  
            closest_point          = ref_xy[distances.argmin(axis=1)[0]]

            manually_chosen_ = np.argsort(np.mean(distances, axis=0))[:2]

            gjcenter_index = manually_chosen_[0].item()

            if manually_chosen_[0] not in gjcenter_index_list:
                gjcenter_index_list.append(gjcenter_index) 

            elif gjcenter_index in gjcenter_index_list:
                gjcenter_index = manually_chosen_[1].item()
                gjcenter_index_list.append(gjcenter_index)

            elif gjcenter_index in gjcenter_index_list:
                gjcenter_index = manually_chosen_[2].item()
                gjcenter_index_list.append(gjcenter_index) 

            self.mapping_gjcenter_id_to_mask_id[gjcenter_index] = mask_index
            self.mapping_mask_id_to_gjcenter_id[mask_index] = gjcenter_index

    def show_mask_order(self, center_geojson_file=None, center_geojson_crs="EPSG:4326", figsize=(20, 15), fontsize=14, alpha=0.5, satellite_image=None):
        if center_geojson_file is not None:
            self.mapping_geojson_center_to_mask_id(center_geojson_file=center_geojson_file, center_geojson_crs=center_geojson_crs)
        else:
            self.mapping_geojson_center_to_mask_id()

        fig, axs = plt.subplots(1, 1, figsize=figsize) 
        if satellite_image is not None:
            axs.imshow(satellite_image.transpose(1,2,0))
        axs.imshow(self.mask.astype(np.uint8), cmap='turbo', alpha=alpha)
        axs.scatter(self.center_pixel_x, self.center_pixel_y, c='cyan', s=10)

        for geojson_center_id, (x, y) in enumerate(zip(self.center_pixel_x, self.center_pixel_y)):
            # Place the text at the coordinates (x[i], y[i])
            # The 'xytext' argument can be used to offset the text from the point
            plt.annotate(text=geojson_center_id, xy=(x, y), xytext=(5, 5), fontsize=fontsize, textcoords='offset points')
            
    def get_a_binary_mask(self, geojson_center_id):
        mask_id = self.mapping_gjcenter_id_to_mask_id[geojson_center_id]
        gray_bf = 1*(self.mask == mask_id) 
        return gray_bf
    
    def update_a_binary_mask(self, new_mask, geojson_center_id, record_list = None):
        mask_2D = self.mask.copy()
        mask_id = self.mapping_gjcenter_id_to_mask_id[geojson_center_id]
        mask_2D[mask_2D == mask_id]  = mask_id*new_mask[mask_2D == mask_id]  

        #mask_2D   = mask_2D.reshape(1, mask_2D.shape[0], mask_2D.shape[1])
        self.mask = mask_2D

        
        if record_list is not None:
            for record_ in record_list:
                record_id = {"gjcenter_id": geojson_center_id}
                merged_dict = {**record_id, **record_}
                self.record_list.append(merged_dict)
        else:
            self.record_list.append({"gjcenter_id": geojson_center_id})

        return mask_2D
    

    def filling_holes(self, gray):
 

        des = cv2.bitwise_not(gray)
        contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des,[cnt],0, 255,-1)

        gray = cv2.bitwise_not(des)

        record_dict = {"Operation": "filling_holes", "Kernel": None}
        
        return gray, record_dict

    def erosion(self, gray, kernel_size=5): 

        # Creating kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Using cv2.erode() method 
        gray = cv2.erode(gray, kernel, iterations=1) 


        record_dict = {"Operation": "erosion", "Kernel": kernel_size} 

        return gray, record_dict

    def dilation(self, gray, kernel_size=5):

        # Creating kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Using cv2.erode() method 
        gray = cv2.dilate(gray, kernel) 


        record_dict = {"Operation": "dilation", "Kernel": kernel_size} 

        return gray, record_dict 