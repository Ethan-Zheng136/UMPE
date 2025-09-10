import mmcv
import numpy as np
import cv2
import os
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadSDRasterFromFile:
    def __init__(self, to_float32=True, normalize=True):
        self.to_float32 = to_float32
        self.normalize = normalize
        self.sdmap_root = '/path/to/your/data'
    
    def __call__(self, results):
        token = results['sample_idx']
        sdmap_filename = os.path.join(self.sdmap_root, f'{token}.png')

        if os.path.exists(sdmap_filename):
            sdmap = cv2.imread(sdmap_filename)
            sdmap = cv2.cvtColor(sdmap, cv2.COLOR_BGR2RGB)
        else:
            sdmap = np.ones((200, 100, 3), dtype=np.uint8) * 255 

        if self.to_float32:
            sdmap = sdmap.astype(np.float32)
        
        if self.normalize:
            sdmap = sdmap / 255.0
        
        results['SDRaster_img'] = sdmap
        return results