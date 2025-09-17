<div align="center">
<h2>Unified Map Prior Encoder for Mapping and Planning</h2>
<p style="font-size: 64px; font-weight: bold; margin-top: 10px;"><strong>ICRA 2026 for consideration</strong></p>

 
Online mapping and end-to-end (E2E) planning
in autonomous driving are still largely sensor-centric, leaving
rich map priors (HD/SD vector maps, rasterized SD maps,
and satellite imagery) underused due to heterogeneity, pose
drift, and inconsistent availability at test time. We present
UMPE, a Unified Map Prior Encoder that can ingest any
subset of four priors and fuse them with BEV features for
both mapping and planning. UMPE has two branches. The
vector encoder pre-aligns HD/SD polylines with a frame-wise
SE(2) correction, encodes points via multi-frequency sinusoidal
features, and produces polyline tokens with confidence scores.
BEV queries then apply cross-attention with confidence bias,
followed by normalized channel-wise gating to avoid length im-
balance and to softly down-weight uncertain sources. The raster
encoder shares a ResNet-18 backbone conditioned by FiLM
(scaling/shift at every stage), performs SE(2) micro-alignment,
and injects priors through zero-initialized residual fusion so
the network starts from a do-no-harm baseline and learns
to add only useful prior evidence. A vector-then-raster fusion
order reflects the inductive bias of “geometry first, appearance
second.” On nuScenes mapping, UMPE lifts MapTRv2 from
61.5 → 67.4 mAP (+5.9) and MapQR from 66.4 → 71.7 mAP
(+5.3). On Argoverse2, UMPE adds +4.1 mAP over strong
baselines. UMPE is compositional: when trained with all priors,
it outperforms single-prior models even when only one prior
is available at test time, demonstrating powerset robustness.
For E2E planning (VAD backbone, nuScenes), UMPE reduces
trajectory error from 0.72 → 0.42 m L2 (avg. −0.30 m) and
collision rate from 0.22% → 0.12% (−0.10%), surpassing recent
prior-injection methods. These results show that a unified,
alignment-aware treatment of heterogeneous map priors yields
better mapping and better planning. 

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