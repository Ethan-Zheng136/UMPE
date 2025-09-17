<div align="center">
<h2>Unified Map Prior Encoder for Mapping and Planning</h2>
<p style="font-size: 64px; font-weight: bold; margin-top: 10px;"><strong>ICRA 2026 for consideration</strong></p>

<table>
<tr>
<td width="60%">
Autonomous driving systems underuse rich map priors (HD/SD vectors, rasterized SD maps, satellite imagery) due to heterogeneity and pose drift. We present UMPE, a Unified Map Prior Encoder that ingests any subset of four map types and fuses them with BEV features. UMPE uses a vector encoder with SE(2) alignment and confidence scoring, plus a raster encoder with FiLM conditioning and zero-initialized residual fusion. On nuScenes mapping, UMPE improves MapTRv2 61.5 → 67.4 mAP (+5.9) and MapQR 66.4 → 71.7 mAP (+5.3). For E2E planning, it reduces trajectory error from 0.72 → 0.42 m L2 (avg. −0.30 m) and collision rate from 0.22% → 0.12% (−0.10%). These results show that a unified,
alignment-aware treatment of heterogeneous map priors yields better mapping and better planning
</td>
<td width="40%">
<img src="assets/teaser.png" alt="UMPE teaser" style="width: 40%; height: auto;">

## Overview
![pipeline](assets/method.png)

We first estimate map elements online by encoding multi-view images into a common BEV feature space
to regress map element vertices. Each vertex’s uncertainty is modeled using our proposed Covariance-based Uncertainty method, which
leverages 2D Gaussian distribution templates. This uncertainty information, along with the original map vertices, is then passed to the
downstream trajectory prediction module, which operates in two parallel streams: one that incorporates uncertainty and one that does not.
Finally, the proposed Proprioceptive Scenario Gating (MLP network) dynamically adapts the optimal trajectory prediction based on the
initial future trajectories prediction from these two streams.

## Our results

![Main_Table](assets/Main_Table.png)

## Our demo video
Click the cover image to watch the HD demo on YouTube.
[![Watch the video](assets/video_cover.png)](https://youtu.be/SbicP4tTv7I)

## Getting Started
- [Environment Setup](docs/env.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Mapping Train and Eval](docs/map.md)
- [Merge Map and Trajectory Dataset](docs/adaptor.md)
- [Trajectory Train and Eval](docs/trj.md)
- [Visualization](docs/visualization.md)

## Checkpoints
Our trajectory prediction checkpoints are [here](https://drive.google.com/drive/folders/1npxVMMCyMgckBBXUnuRW8M3sYexpAObd?usp=sharing).

## Dataset

All the trajectory prediction data（for `MapTR`, `StreamMapNet`, `MapTRv2` and `MapTRv2 CL`）can be generated using our future checkpoints, with a total size of approximately 600GB.
Dataset Structure is as follows:
```
DelvingUncPrediction
├── trj_data/
│   ├── maptr/
│   |   ├── train/
│   |   |   ├── data/
│   |   |   |   ├── scene-{scene_id}.pkl
│   |   ├── val/
│   ├── maptrv2/
│   ├── maptrv2_CL/
│   ├── stream/
```

## Catalog

- [x] Code release
  - [x] MapTR
  - [x] MapTRv2
  - [x] StreamMapNet
  - [x] HiVT
  - [x] DenseTNT
- [x] Visualization Code
- [x] Untested version released + Instructions
- [x] Initialization




## License

This repository is licensed under [Apache 2.0](LICENSE).


## Citation
If you find this project useful, feel free to cite our work!

```bibtex
@article{zhang2025delving,
  title={Delving into Mapping Uncertainty for Mapless Trajectory Prediction},
  author={Zhang, Zongzheng and Qiu, Xuchong and Zhang, Boran and Zheng, Guantian and Gu, Xunjiang and Chi, Guoxuan and Gao, Huan-ang and Wang, Leichen and Liu, Ziming and Li, Xinrun and others},
  journal={arXiv preprint arXiv:2507.18498},
  year={2025}
}
```