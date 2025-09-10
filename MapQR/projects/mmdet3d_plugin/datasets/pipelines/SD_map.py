import mmcv
import numpy as np
import cv2
import os
from mmdet.datasets.builder import PIPELINES

import cv2
import os
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadSDRasterFromFile:
    def __init__(self, to_float32=True, normalize=True):
        self.to_float32 = to_float32
        self.normalize = normalize
        # self.sdmap_root = '/DATA_EDS2/zhenggt2407/VAD_pro/VAD/data/nuscenes/vad_train_sdmap_raster_vis'
        self.sdmap_root = '/DATA_EDS2/zhenggt2407/MapTR-maptrv2/data/nuscenes/sd_map_reraster2_vis'
    
    def __call__(self, results):
        # index = results['index']
        # sdmap_filename = os.path.join(self.sdmap_root, f'sample_{index+1:06d}.png')
        token = results['sample_idx']
        sdmap_filename = os.path.join(self.sdmap_root, f'{token}.png')

        if os.path.exists(sdmap_filename):
            sdmap = cv2.imread(sdmap_filename)
            sdmap = cv2.cvtColor(sdmap, cv2.COLOR_BGR2RGB)
            # print('---------------------yes---------------------')
        else:
            sdmap = np.ones((200, 100, 3), dtype=np.uint8) * 255  # 白色图像
            # print('---------------------no---------------------')
            # print(f"File not found: {sdmap_filename}, using white image")
        if self.to_float32:
            sdmap = sdmap.astype(np.float32)
        
        if self.normalize:
            sdmap = sdmap / 255.0
        
        results['SDRaster_img'] = sdmap
        return results