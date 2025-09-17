## Visualize prediction

### MapTRv2 & MapQR
```shell
cd /path/to/MapTRv2/
# visualize nuscenes dataset
python tools/maptrv2/nusc_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt

#visualize argoverse2 dataset
python tools/maptrv2/av2_vis_pred.py /path/to/experiment/config /path/to/experiment/ckpt
```


### VAD

```shell
cd /path/to/VAD/
conda activate vad
python tools/analysis_tools/visualization.py --result-path /path/to/inference/results --save-path /path/to/save/visualization/results
```

The inference results is a prefix_results_nusc.pkl automaticly saved to the work_dir after running evaluation. It's a list of prediction results for each validation sample.