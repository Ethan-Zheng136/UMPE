# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test


**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MapTRv2 with 8 GPUs 
```
cd /path/to/MapTRv2
./tools/dist_train.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep_umpe.py 8
```

Eval MapTRv2
```
cd /path/to/MapTRv2
./tools/dist_test_map.sh ./projects/configs/maptrv2/maptrv2_nusc_r50_24ep_umpe.py ./path/to/ckpts.pth
```

Train MapQR with 8 GPUs 
```
cd /path/to/MapQR
./tools/dist_train.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep_new_umpe.py 8
```

Eval MapQR
```
cd /path/to/MapQR
./tools/dist_test_map.sh ./projects/configs/mapqr/mapqr_nusc_r50_24ep_new_umpe.py ./path/to/ckpts.pth
```




# Visualization 

we provide tools for visualization and benchmark under `path/to/MapQR/tools/maptrv2`
