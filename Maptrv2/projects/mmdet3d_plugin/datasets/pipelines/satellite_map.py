import mmcv
import numpy as np
import cv2
import os
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadSatelliteFromFile:
    def __init__(self, to_float32=True, normalize=True):
        self.to_float32 = to_float32
        self.normalize = normalize
        self.satellite_root = "/path/to/your/data"
    
    def __call__(self, results):
        timestamp = results['timestamp']
        map_location = results['map_location']
        
        pattern = f"{timestamp}_{map_location}_*.jpg"
        filepath = os.path.join(self.satellite_root, pattern)

        import glob
        files = glob.glob(filepath)
        satellite_filepath = files[0] 
        
        satellite_img = cv2.imread(satellite_filepath)
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        satellite_img = cv2.resize(satellite_img, (100, 200))
        
        if self.to_float32:
            satellite_img = satellite_img.astype(np.float32)

        if self.normalize:
            satellite_img = satellite_img / 255.0
        
        results['satellite_img'] = satellite_img
        return results