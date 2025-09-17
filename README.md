<div align="center">
<h2>Unified Map Prior Encoder for Mapping and Planning</h2>
<p style="font-size: 64px; font-weight: bold; margin-top: 10px;"><strong>ICRA 2026 for consideration</strong></p>

Autonomous driving systems underuse rich map priors (HD/SD vectors, raster maps, satellite imagery) due to heterogeneity and pose drift. We present UMPE, a Unified Map Prior Encoder that ingests any subset of four map types and fuses them with BEV features. UMPE has two branches. The vector encoder pre-aligns HD/SD polylines with a frame-wise SE(2) correction, encodes points via multi-frequency sinusoidal
features, and produces polyline tokens with confidence scores.
BEV queries then apply cross-attention with confidence bias,
followed by normalized channel-wise gating to avoid length im-
balance and to softly down-weight uncertain sources. The raster
encoder shares a ResNet-18 backbone conditioned by FiLM
(scaling/shift at every stage), performs SE(2) micro-alignment,
and injects priors through zero-initialized residual fusion so
the network starts from a do-no-harm baseline and learns
to add only useful prior evidence. On nuScenes mapping, UMPE improves MapTRv2 by +5.9 mAP and MapQR by +5.3 mAP. For E2E planning, it reduces trajectory error by 0.30m and collision rate by 0.10%. UMPE shows powerset robustness: models trained with all priors outperform single-prior models even when only one prior is available at test time.

The vector encoder pre-aligns HD/SD polylines with a frame-wise
SE(2) correction, encodes points via multi-frequency sinusoidal
features, and produces polyline tokens with confidence scores.
BEV queries then apply cross-attention with confidence bias,
followed by normalized channel-wise gating to avoid length im-
balance and to softly down-weight uncertain sources. The raster
encoder shares a ResNet-18 backbone conditioned by FiLM
(scaling/shift at every stage), performs SE(2) micro-alignment,
and injects priors through zero-initialized residual fusion so
the network starts from a do-no-harm baseline and learns
to add only useful prior evidence. 

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