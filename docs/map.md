# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test


**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTRv2 with 8 GPUs 
```
cd /path/to/MapTRv2
./tools/dist_train.sh ./projects/configs/maptr/maptr_nsuc_r50_24_UMPE_.py 8
```

Eval MapTRv2
```
cd /path/to/MapTRv2
./tools/dist_test_map.sh ./projects/configs/maptr/maptr_nusc_r50_24e_UMPE.py ./path/to/ckpts.pth
```

Train MapQR with 8 GPUs 
```
cd /path/to/MapQR
./tools/dist_train.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep_new_UMPE.py 8
```

Eval MapQR
```
cd /path/to/MapQR
./tools/dist_test_map.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep_new_UMPE.py ./path/to/ckpts.pth
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapQR/tools/maptrv2`