<div align="left">
<div align="center">
<h2>Unified Map Prior Encoder for Mapping and Planning</h2>
<p style="font-size: 64px; font-weight: bold; margin-top: 10px;"><strong>Submit ICRA 2026 for consideration</strong></p>
</div>

</td>
</tr>
</table>

<div align="left">
<img align="right" src="assets/teaser.png" width="300">

Autonomous driving systems underuse rich map priors (HD/SD vectors, rasterized SD maps, satellite imagery) due to heterogeneity and pose drift. We present UMPE, a Unified Map Prior Encoder that ingests any subset of four map types and fuses them with BEV features. UMPE uses a vector encoder with SE(2) alignment and confidence scoring, plus a raster encoder with FiLM conditioning and zero-initialized residual fusion. On nuScenes mapping, UMPE improves MapTRv2 61.5 → 67.4 mAP (+5.9) and MapQR 66.4 → 71.7 mAP (+5.3). For E2E planning, it reduces trajectory error from 0.72 → 0.42 m L2 (avg. −0.30 m) and collision rate from 0.22% → 0.12% (−0.10%). These results show that a unified, alignment-aware treatment of heterogeneous map priors yields better mapping and better planning.

</div>

## Overview
![pipeline](assets/method.png)

<div align="left">
UMPE has two branches. The
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
second.”
</div>

## Our results

![mainable](assets/maintable.jpg)

<!-- ## Our demo video
Click the cover image to watch the HD demo on YouTube.
[![Watch the video](assets/video_cover.png)](https://youtu.be/SbicP4tTv7I) -->

## Getting Started
- [Environment Setup](docs/env.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Mapping Train and Eval](docs/map.md)
- [E2E planning Train and Eval](docs/planning.md)
- [Visualization](docs/vis.md)


## Checkpoints
Our trajectory prediction checkpoints are [here](https://drive.google.com/drive/folders/1npxVMMCyMgckBBXUnuRW8M3sYexpAObd?usp=sharing).

## Dataset

All the trajectory prediction data（for `MapTR`, `StreamMapNet`, `MapTRv2` and `MapTRv2 CL`）can be generated using our future checkpoints, with a total size of approximately 600GB.
Dataset Structure is as follows:
```
UMPE
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
│   ├── argoverse2/
│   │   ├── sensor/
|   |   |   |—— train/
|   |   |   |—— val/
|   |   |   |—— test/
```

## Catalog

<!-- - [x] Code release
  - [x] MapTR
  - [x] MapTRv2
  - [x] StreamMapNet
  - [x] HiVT
  - [x] DenseTNT
- [x] Visualization Code
- [x] Untested version released + Instructions
- [x] Initialization -->




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