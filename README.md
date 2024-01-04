# DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-98E4FF.svg)](https://openreview.net/attachment?id=CY1xatvEQj&name=pdf) 
[![arXiv](https://img.shields.io/badge/arXiv-2306.14685-b31b1b.svg)](https://arxiv.org/abs/2306.14685)
[![website](https://img.shields.io/badge/website-Gitpage-yellow)](https://ximinng.github.io/DiffSketcher-project/)

This repository contains our official implementation of the NeurIPS 2023 paper: DiffSketcher: Text Guided Vector Sketch
Synthesis through Latent Diffusion Models, which can generate high-quality vector sketches based on text prompts. Our
project page can be found [here](https://ximinng.github.io/DiffSketcher-project/).

![title](./img/title.png)

## :new: Update

- [01/2024] 🔥 **We released the [SVGDreamer](https://ximinng.github.io/SVGDreamer-project/). SVGDreamer is
  a novel text-guided vector graphics synthesis method. This method considers both the editing of vector graphics and
  the quality of the synthesis.**
- [12/2023] 🔥 **We released the [PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender). Pytorch-SVGRender is
  the go-to library for state-of-the-art differentiable rendering methods for image vectorization.**
- [11/2023] We thank [@camenduru](https://github.com/camenduru) for implementing
  the [DiffSketcher-colab](https://github.com/camenduru/DiffSketcher-colab).
- [10/2023] We released the DiffSketcher code.
- [10/2023] We released the [VectorFusion code](https://github.com/ximinng/VectorFusion-pytorch).

### TODO

- [ ] Add a webUI demo.
- [x] Add support for colorful results and oil painting.

## :wrench: Installation

Create a new conda environment:

```shell
conda create --name diffsketcher python=3.10
conda activate diffsketcher
```

Install pytorch and the following libraries:

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install omegaconf BeautifulSoup4
pip install opencv-python scikit-image matplotlib visdom wandb
pip install triton numba
pip install numpy scipy timm scikit-fmm einops
pip install accelerate transformers safetensors datasets
```

Install CLIP:

```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Install diffusers:

```shell
pip install diffusers==0.20.2
```

Install xformers (require `python=3.10`):

```shell
conda install xformers -c xformers
```

Install diffvg:

```shell
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
pip install svgwrite svgpathtools cssutils torch-tools
python setup.py install
```

## 🔥 Quickstart

### Example:

Preview:

![sydney_opera_house](./img/sydney_opera_house.svg)

Script:

```shell
python run_painterly_render.py \ 
  -c diffsketcher.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=96 sds.warmup=1000 num_iter=1500" \ 
  -pt "a photo of Sydney opera house" \ 
  -respath ./workdir/sydney_opera_house \ 
  -d 8019 \
  --download
```

- `-c` a.k.a `--config`: configuration file, saving in `DiffSketcher/config/`.
- `-eval_step`: the step size used to eval the method (**too frequent calls will result in longer times**).
- `-save_step`: the step size used to save the result (**too frequent calls will result in longer times**).
- `-update`: a tool for editing the hyper-params of the configuration file, so you don't need to create a new yaml.
- `-pt` a.k.a `--prompt`: text prompt.
- `-respath` a.k.a `--results_path`: the folder to save results.
- `-d` a.k.a `--seed`: random seed.
- `--download`: download models from huggingface automatically **when you first run them**.

crucial:

- `-update "token_ind=2"` indicates the index of cross-attn maps to init strokes.
- `-update "num_paths=96"` indicates the number of strokes.

optional:

- `-npt`, a.k.a `--negative_prompt`: negative text prompt.
- `-mv`, a.k.a `--make_video`: make a video of the rendering process (**it will take much longer**).
- `-frame_freq`, a.k.a `--video_frame_freq`: control video frame.
- **Note:** [Download](https://huggingface.co/akhaliq/CLIPasso/blob/main/u2net.pth) U2Net model and place
  in `checkpoint/` dir if `xdog_intersec=True`
- add `enable_xformers=True` in `-update` to enable xformers for speeding up.
- add `gradient_checkpoint=True` in `-update` to use gradient checkpoint for low VRAM.

### Another example

Preview:

![sydney_opera_house](./img/sydney_opera_house_width.svg)

Script:

```shell
python run_painterly_render.py \ 
  -c diffsketcher-width.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=48 num_iter=500" \ 
  -pt "a photo of Sydney opera house" \ 
  -respath ./workdir/sydney_opera_house \ 
  -d 8019 \
  --download
```

### More Sketch Examples

**check the [run.md](https://github.com/ximinng/DiffSketcher/blob/main/run.md) for more scripts.**

### Oil Painting

Preview:

![portrait_woman_oil_painting](./img/portrait_woman_oil_painting.svg)

Script:

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=2000 num_iter=1500" \
  -pt "portrait of latin woman having a spiritual awaking, eyes closed, slight smile, illuminating lights, oil painting, by Van Gogh" \
  -respath ./workdir/portrait_woman_oil_painting \
  -d 998
 
CUDA_VISIBLE_DEVICES=1 python run_painterly_render.py -c diffsketcher-color.yaml -eval_step 10 -save_step 10 -update "token_ind=4 num_paths=2000 num_iter=3000" -pt "portrait of latin woman having a spiritual awaking, eyes closed, slight smile, illuminating lights, oil painting, by Van Gogh" -respath ./workdir/portrait_oil -rdbz
```

### Colorful Results

Preview:

![horse_rgba](./img/horse_rgba.svg)

Script:

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=2 num_paths=128 num_iter=1500" \
  -pt "A horse is drinking water by the lake" \
  -respath ./workdir/draw_horse_rgba \
  -d 998
```

## :books: Acknowledgement

The project is built based on the following repository:

- [BachiLi/diffvg](https://github.com/BachiLi/diffvg)
- [yael-vinker/CLIPasso](https://github.com/yael-vinker/CLIPasso)
- [huggingface/diffusers](https://github.com/huggingface/diffusers)

We gratefully thank the authors for their wonderful works.

## :paperclip: Citation

If you use this code for your research, please cite the following work:

```
@inproceedings{xing2023diffsketcher,
    title={DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models},
    author={XiMing Xing and Chuang Wang and Haitao Zhou and Jing Zhang and Qian Yu and Dong Xu},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=CY1xatvEQj}
}
```

## :copyright: Licence

This work is licensed under a MIT License.