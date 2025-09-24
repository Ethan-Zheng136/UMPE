import mmcv
import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES


# @PIPELINES.register_module()
# class LoadSatelliteFromFile:
#     def __init__(self, to_float32=True, normalize=True):
#         self.to_float32 = to_float32
#         self.normalize = normalize
    
#     def __call__(self, results):
#         satellite_filename = results['satellite_filename']
        
#         satellite_img = cv2.imread(satellite_filename)
#         satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
#         satellite_img = cv2.resize(satellite_img, (100, 200))
        
#         if self.to_float32:
#             satellite_img = satellite_img.astype(np.float32)

#         if self.normalize:
#             satellite_img = satellite_img / 255.0
        
#         results['satellite_img'] = satellite_img
#         return results

import mmcv
import numpy as np
import cv2
import os
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadSatelliteFromFile:
    def __init__(self, to_float32=True, bev_h=200, bev_w=200, normalize=True):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.to_float32 = to_float32
        self.normalize = normalize
        self.satellite_root = "/share/gha/shared/VAD/prior/rotated_trainval"
    
    def __call__(self, results):
        timestamp = results['real_timestamp']
        map_location = results['map_location']
        # print(results['index'])
        
        # print('timestamp:', timestamp)
        pattern = f"{timestamp}_*.jpg"
        filepath = os.path.join(self.satellite_root, pattern)

        import glob
        files = glob.glob(filepath)
        # print('files:', files)
        satellite_filepath = files[0]
        
        satellite_img = cv2.imread(satellite_filepath)
        satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_BGR2RGB)
        satellite_img = cv2.resize(satellite_img, (self.bev_h, self.bev_w))
        
        if self.to_float32:
            satellite_img = satellite_img.astype(np.float32)

        if self.normalize:
            satellite_img = satellite_img / 255.0
        
        results['satellite_img'] = satellite_img
        return results