# DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models

[![NeurIPS 2023](https://img.shields.io/badge/NeurIPS%202023-Paper-6420AA?style=for-the-badge&logo=openreview&logoColor=white)](https://openreview.net/attachment?id=CY1xatvEQj&name=pdf) [![ArXiv](https://img.shields.io/badge/arXiv-2306.14685-FF6347?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2306.14685) [![Project Website](https://img.shields.io/badge/Website-Project%20Page-4682B4?style=for-the-badge&logo=github&logoColor=white)](https://ximinng.github.io/DiffSketcher-project/) [![Demo](https://img.shields.io/badge/Demo-Gradio-00CED1?style=for-the-badge&logo=gradio&logoColor=white)](https://huggingface.co/spaces/SVGRender/DiffSketcher)

This repository contains our official implementation of the NeurIPS 2023 paper: DiffSketcher: Text Guided Vector Sketch
Synthesis through Latent Diffusion Models, which can generate high-quality vector sketches based on text prompts.

![teaser1](./img/teaser1.png)
![teaser2](./img/teaser2.png)

**DiffSketcher Rendering Process:**

| <img src="./img/0.gif" style="width: 200px; height: 200px;">            | <img src="./img/1.gif" style="width: 200px; height: 200px;">                | <img src="./img/2.gif" style="width: 200px; height: 200px;"> |
|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------|
| Prompt: Macaw full color, ultra detailed, realistic, insanely beautiful | Prompt: Very detailed masterpiece painting of baby yoda hoding a lightsaber | Prompt: Sailboat sailing in the sea on a clear day           |

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

## 📌 Installation Guide

To quickly get started with **DiffSketcher**, follow the steps below.  
These instructions will help you run **quick inference locally**.

#### 🚀 **Option 1: Standard Installation**

Run the following command in the **top-level directory**:

```shell
chmod +x script/install.sh
bash script/install.sh
```

#### 🐳 Option 2: Using Docker

```shell
chmod +x script/run_docker.sh
sudo bash script/run_docker.sh
```

## 🔥 Quickstart

### Case: Sydney Opera House

**Preview:**

|                                    Attention Map                                     |                                   Control Points Init                                   |                                Strokes Initialization                                 |                                        100 step                                         |                                          500 step                                           |
|:------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| <img src="./img/SydneyOperaHouse/attn-map.png" style="width: 200px; height: 200px;"> | <img src="./img/SydneyOperaHouse/points-init.png" style="width: 200px; height: 200px;"> | <img src="./img/SydneyOperaHouse/svg_iter0.svg" style="width: 200px; height: 200px;"> | <img src="./img/SydneyOperaHouse/svg_iter100.svg" style="width: 200px; height: 200px;"> | <img src="./img/SydneyOperaHouse/visual_best_96P.svg" style="width: 200px; height: 200px;"> |

**From the abstract to the concrete:**

|                        16 Paths                        |                        36 Paths                        |                        48 Paths                        |                        96 Paths                        |                        128 Paths                        |
|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:------------------------------------------------------:|:-------------------------------------------------------:|
| <img src="./img/SydneyOperaHouse/visual_best_16P.svg"> | <img src="./img/SydneyOperaHouse/visual_best_36P.svg"> | <img src="./img/SydneyOperaHouse/visual_best_48P.svg"> | <img src="./img/SydneyOperaHouse/visual_best_96P.svg"> | <img src="./img/SydneyOperaHouse/visual_best_128P.svg"> |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=96 num_iter=800" \
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

**crucial:**

- `-update "token_ind=4"` indicates the index of cross-attn maps to init strokes.
- `-update "num_paths=96"` indicates the number of strokes.

**optional:**

- `-npt`, a.k.a `--negative_prompt`: negative text prompt.
- `-mv`, a.k.a `--make_video`: make a video of the rendering process (**it will take much longer**).
- `-frame_freq`, a.k.a `--video_frame_freq`: the interval of the number of steps to save the image.
- `-framerate`, a.k.a `--video_frame_rate`: control the playback speed of the output video.
- **Note:** [Download](https://huggingface.co/akhaliq/CLIPasso/blob/main/u2net.pth) U2Net model and place
  in `checkpoint/` dir if `xdog_intersec=True`
- add `enable_xformers=True` in `-update` to enable xformers for speeding up.
- add `gradient_checkpoint=True` in `-update` to use gradient checkpoint for low VRAM.

### Case: Sydney Opera House in ink painting style

**Preview:**

| <img src="./img/SydneyOperaHouse-ink/svg_iter0.svg"> | <img src="./img/SydneyOperaHouse-ink/svg_iter100.svg"> | <img src="./img/SydneyOperaHouse-ink/svg_iter200.svg"> | <img src="./img/SydneyOperaHouse-ink/visual_best.svg"> |
|------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| Strokes Initialization                               | 100 step                                               | 200 step                                               | 990 step                                               |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-width.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=48 num_iter=800" \
  -pt "a photo of Sydney opera house" \
  -respath ./workdir/sydney_opera_house_ink \
  -d 8019 \
  --download
```

### Oil Painting

**Preview:**

| <img src="./img/LatinWomanPortrait/svg_iter0.svg"> | <img src="./img/LatinWomanPortrait/svg_iter100.svg"> | <img src="./img/LatinWomanPortrait/visual_best.svg"> |
|----------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Strokes Initialization                             | 100 step                                             | 570 step                                             |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=5 num_paths=1000 num_iter=1000 guidance_scale=7.5" \
  -pt "portrait of latin woman having a spiritual awaking, eyes closed, slight smile, illuminating lights, oil painting, by Van Gogh" \
  -npt "text, signature, title, heading, watermark, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck" \
  -respath ./workdir/latin_woman_portrait -d 58548
```

**Preview:**

| <img src="./img/WomanWithCrown/svg_iter0.svg"> | <img src="./img/WomanWithCrown/svg_iter100.svg"> | <img src="./img/WomanWithCrown/visual_best.svg"> |
|------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Strokes Initialization                         | 100 step                                         | 570 step                                         |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=5 num_paths=1000 num_iter=1000 guidance_scale=7.5" \
  -pt "a painting of a woman with a crown on her head, art station front page, dynamic portrait style, many colors in the background, olpntng style, oil painting, forbidden beauty" \
  -npt "2 heads, 2 faces, cropped image, out of frame, draft, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb, disfigured, cut-off, ugly, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, disgusting, poorly drawn, mutilated, mangled, extra fingers, duplicate artifacts, morbid, gross proportions, missing arms, mutated hands, mutilated hands, cloned face, malformed, blur haze" \
  -respath ./workdir/woman_with_crown -d 178351
```

**Preview:**

| <img src="./img/BeautifulGirl_OilPainting/svg_iter0.svg"> | <img src="./img/BeautifulGirl_OilPainting/svg_iter100.svg"> | <img src="./img/BeautifulGirl_OilPainting/visual_best.svg"> |
|-----------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|
| Strokes Initialization                                    | 100 step                                                    | 420 step                                                    |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=5 num_paths=1000 num_iter=1000 guidance_scale=7.5" \
  -pt "a painting of a woman with a crown on her head, art station front page, dynamic portrait style, many colors in the background, olpntng style, oil painting, forbidden beauty" \
  -npt "2 heads, 2 faces, cropped image, out of frame, draft, deformed hands, twisted fingers, double image, malformed hands, multiple heads, extra limb, ugly, poorly drawn hands, missing limb, disfigured, cut-off, ugly, grain, low-res, Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, floating limbs, disconnected limbs, disgusting, poorly drawn, mutilated, mangled, extra fingers, duplicate artifacts, morbid, gross proportions, missing arms, mutated hands, mutilated hands, cloned face, malformed, blur haze" \
  -respath ./workdir/woman_with_crown -d 178351
```

### Colorful Results

**Preview:**

| <img src="./img/castle-rgba/svg_iter0.svg"> | <img src="./img/castle-rgba/svg_iter100.svg"> | <img src="./img/castle-rgba/visual_best.svg"> |
|---------------------------------------------|-----------------------------------------------|-----------------------------------------------|
| Strokes Initialization                      | 100 step                                      | 340 step                                      |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=5 num_paths=1000 num_iter=800 guidance_scale=7" \
  -pt "a beautiful snow-covered castle, a stunning masterpiece, trees, rays of the sun, Leonid Afremov" \
  -npt "poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face" \
  -respath ./workdir/castle -d 370880
```

**Preview:**

| <img src="./img/castle-rgba-2/svg_iter0.svg"> | <img src="./img/castle-rgba-2/svg_iter100.svg"> | <img src="./img/castle-rgba-2/visual_best.svg"> |
|-----------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Strokes Initialization                        | 100 step                                        | 850 step                                        |

**Script:**

```shell
python run_painterly_render.py \
  -c diffsketcher-color.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=5 num_paths=1000 num_iter=800 guidance_scale=7" \
  -pt "a beautiful snow-covered castle, a stunning masterpiece, trees, rays of the sun, Leonid Afremov" \
  -npt "poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face" \
  -respath ./workdir/castle -d 478376
```

### DiffSketcher + Style Transfer

**Preview:**

| <img src="./img/FrenchRevolution-ST/generated.png" style="width: 250px; height: 250px;"> | <img src="./img/starry.jpg" style="width: 250px; height: 250px;"> | <img src="./img/FrenchRevolution-ST/french_ST.svg" style="width: 250px; height: 250px;"> |
|------------------------------------------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| Generated sample                                                                         | Style Image                                                       | Result                                                                                   |

**Script:**

```shell
python run_painterly_render.py \
  -tk style-diffsketcher -c diffsketcher-style.yaml \
  -eval_step 10 -save_step 10 \
  -update "token_ind=4 num_paths=2000 style_warmup=0 style_strength=1 softmax_temp=0.4 sds.grad_scale=0 lr_scheduler=True num_iter=2000" \
  -pt "The French Revolution, highly detailed, 8k, ornate, intricate, cinematic, dehazed, atmospheric, oil painting, by Van Gogh" \
  -style ./img/starry.jpg \
  -respath ./workdir/style_transfer \
  -d 876809
```

- `-style`: the path of style img place.
- `style_warmup`:  add style loss after `style_warmup` step.
- `style_strength`:  How strong the style should be. 100 (max) is a lot. 0 (min) is no style.

### More Sketch Results

**check the [Examples.md](https://github.com/ximinng/DiffSketcher/blob/main/Examples.md) for more cases.**

### TODO

- [x] Add a webUI demo.
- [x] Add support for colorful results and oil painting.

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