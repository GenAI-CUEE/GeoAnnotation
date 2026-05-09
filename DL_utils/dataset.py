import os
import tifffile
from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np

class OurDataset(Dataset):
    """Dynamically pairs images and masks and forces masks into strict binary."""
    def __init__(self, data_dir,data_mode = "theos",transform_image=None, transform_mask = None):
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.image_paths = []
        self.mask_paths = []
        self.data_mode = data_mode
        
        # Look for everything ending in "_google"
        if data_mode == "theos":
            search_pattern = os.path.join(data_dir, '*_theos.*')
        else:
            search_pattern = os.path.join(data_dir, '*_google.*')
        google_files = sorted(glob.glob(search_pattern))
        
        for img_path in google_files:
            # Swap "_google" for "_masks" to find the partner file
            if data_mode == "theos":
                mask_path = img_path.replace('_theos', '_masks')
            else:
                mask_path = img_path.replace('_google', '_masks')
            
            # Only add them to the list if BOTH files exist
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                print(f"⚠️ Warning: Could not find matching mask for {img_path}")
                
    def __getitem__(self, index):
        # 1. Read the scientific TIFF using the professor's method (returns a NumPy array)
        if self.data_mode == "theos":
            img_array = tifffile.imread(self.image_paths[index])
        else:
            image = Image.open(self.image_paths[index]).convert("RGB")
            
        mask = Image.open(self.mask_paths[index]).convert("L")

        # 2. Convert the NumPy array back into a PIL Image so your transforms don't crash!
        if self.data_mode == 'theos':
            image = Image.fromarray(img_array[:,:,:3].astype(np.uint8)).convert("RGB")

        # 3. Apply your normal resizing and tensor conversions
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_mask:    
            mask = self.transform_mask(mask)
            
        # 4. Strict binary conversion
        #mask = np.asarray(mask) 
        mask = (mask > 0.0).float()
        
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)