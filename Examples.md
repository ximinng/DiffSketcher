# Qualitative Results

### Case: Horse Drinking Water

**JVSP + ASDS fine-tuning (horse)**

**Preview:**

| canvas size / Rendering | Strokes Initialization                                  | 100 step                                                  |                        visual best                        |
|-------------------------|---------------------------------------------------------|-----------------------------------------------------------|:---------------------------------------------------------:|
| `canvas_size=224`       | <img src="./img/HorseDrinkingWater/svg_iter0_C224.svg"> | <img src="./img/HorseDrinkingWater/svg_iter100_C224.svg"> | <img src="./img/HorseDrinkingWater/visual_best_C224.svg"> |
| `canvas_size=600`       | <img src="./img/HorseDrinkingWater/svg_iter0_C600.svg"> | <img src="./img/HorseDrinkingWater/svg_iter100_C600.svg"> | <img src="./img/HorseDrinkingWater/visual_best_C600.svg"> |

**Script:**

```shell
# canvas_size: 224
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "token_ind=2 num_paths=96" -pt "A horse is drinking water by the lake" -respath ./workdir/draw_horse -d 998
# canvas_size: 600
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "token_ind=2 num_paths=96 image_size=600 width=3.5" -pt "A horse is drinking water by the lake" -respath ./workdir/draw_horse -d 998
```

**train from scratch via ASDS loss + SDXL (horse)**

```shell
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "image_size=1110 token_ind=2 num_paths=96 sds.grad_scale=2 sds.warmup=0 sds.crop_size=1024 clip.vis_loss=0 perceptual.coeff=0 opacity_delta=0.2 num_iter=2000 model_id=sdxl" -pt "A horse is drinking water by the lake" -respath ./workdir/draw_horse -d 998 --download
```

**JVSP + ASDS fine-tune (horse) + including width**

**Preview:**

| <img src="./img/HorseDrinkingWater-ink/svg_iter0.svg"> | <img src="./img/HorseDrinkingWater-ink/svg_iter100.svg"> | <img src="./img/HorseDrinkingWater-ink/svg_iter1000.svg"> |
|--------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|
| Strokes Initialization                                 | 100 step                                                 | 1000 step                                                 |

**Script:**

```shell
python run_painterly_render.py -c diffsketcher-width.yaml -eval_step 10 -save_step 10 -update "token_ind=2 num_paths=96 num_iter=1000 grad_scale=0" -pt "A horse is drinking water by the lake" -respath ./workdir/draw_horse_ink -d 998
```

### Case: 3D Style Sketch

**Preview:**

| <img src="./img/3D_rose/svg_iter0.svg"> | <img src="./img/3D_rose/svg_iter100.svg"> | <img src="./img/3D_rose/visual_best.svg"> |
|-----------------------------------------|-------------------------------------------|-------------------------------------------|
| Strokes Initialization                  | 100 step                                  | 1510 step                                 |

**Script:**

```shell
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "token_ind=4 num_paths=128 sds.grad_scale=0" -pt "A 3d single rose" -respath ./workdir/3d_rose -d 291516 
```

### Case: Elephant

**Preview:**

| <img src="./img/elephant/svg_iter0.svg"> | <img src="./img/elephant/svg_iter100.svg"> | <img src="./img/elephant/semantic_best.svg"> |
|------------------------------------------|--------------------------------------------|----------------------------------------------|
| Strokes Initialization                   | 100 step                                   | 1890 step                                    |

**Script:**

```shell
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "token_ind=2 num_paths=128 softmax_temp=0.4 sds.grad_scale=1 sds.warmup=0 clip.vis_loss=0 perceptual.coeff=0 lr_scheduler=True num_iter=2000 opacity_delta=0.1" -pt "an elephant. minimal 2d line drawing. trending on artstation." -respath ./workdir/elephant -d 197920
```

### Case: Yoda

**Preview:**

| <img src="./img/Yoda/svg_iter0.svg"> | <img src="./img/Yoda/svg_iter100.svg"> | <img src="./img/Yoda/visual_best.svg"> |
|--------------------------------------|----------------------------------------|----------------------------------------|
| Strokes Initialization               | 100 step                               | 1780 step                              |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=7 num_paths=96 sds.warmup=1500 num_iter=2000" \
-pt "Very detailed masterpiece painting of baby yoda hoding a lightsaber, portrait, artstation, concept art by greg rutkowski" \
-respath ./workdir/Yoda \
-d 998
```

### Case: Fox

**Preview:**

| <img src="./img/Fox/svg_iter0.svg"> | <img src="./img/Fox/svg_iter100.svg"> | <img src="./img/Fox/visual_best.svg"> |
|-------------------------------------|---------------------------------------|---------------------------------------|
| Strokes Initialization              | 100 step                              | 1500 step                             |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=2 num_paths=96 softmax_temp=0.5 num_iter=2000" \
-pt "A fox is sitting on the sofa" \
-respath ./workdir/fox \
-d 9007
```

### Case: Balloons

**Preview:**

| <img src="./img/balloons/svg_iter0.svg"> | <img src="./img/balloons/svg_iter100.svg"> | <img src="./img/balloons/visual_best.svg"> |
|------------------------------------------|--------------------------------------------|--------------------------------------------|
| Strokes Initialization                   | 100 step                                   | 2000 step                                  |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=4 num_paths=128 softmax_temp=0.5" \
-pt "Colorful hot air balloons high over the mountains" \
-respath ./workdir/balloons \
-d 9998
```

### Case: A cat on a bicycle

**Preview:**

| <img src="./img/cat_ride_bike/svg_iter0.svg"> | <img src="./img/cat_ride_bike/svg_iter100.svg"> | <img src="./img/cat_ride_bike/visual_best.svg"> |
|-----------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Strokes Initialization                        | 100 step                                        | 1800 step                                       |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py  \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=2 num_paths=48 ssoftmax_temp=0.5 mask_object=False" \
-pt "A cute cat in the style of Pixar animations, wearing a helmet and riding a bike" \
-respath  ./workdir/cat \
-d 8030
```

### Case: Cat

**Preview:**

| <img src="./img/cat/svg_iter0.svg"> | <img src="./img/cat/svg_iter100.svg"> | <img src="./img/cat/final_svg.svg"> |
|-------------------------------------|---------------------------------------|-------------------------------------|
| Strokes Initialization              | 100 step                              | 2000 step                           |

**Script:**

```shell
python run_painterly_render.py -c diffsketcher.yaml -eval_step 10 -save_step 10 -update "token_ind=2 num_paths=128 softmax_temp=0.4 sds.num_aug=4 sds.grad_scale=1 sds.warmup=0 clip.vis_loss=0 perceptual.coeff=0 lr_scheduler=True num_iter=2000 opacity_delta=0.3" -pt "A cat. minimal 2d line drawing. trending on artstation." -respath ./workdir/cat -d 915346
```

### Case: Athens

**Preview:**

| <img src="./img/Athens/svg_iter0.svg"> | <img src="./img/Athens/svg_iter100.svg"> | <img src="./img/Athens/visual_best.svg"> |
|----------------------------------------|------------------------------------------|------------------------------------------|
| Strokes Initialization                 | 100 step                                 | 1380 step                                |

**Script:**

```shell
python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=11 num_paths=140 softmax_temp=0.5 num_iter=2000" \
-pt "A loose ink sketching with watercolors of a modern Athens neighborhood, architectural, detailed, old building and new buildings, quiet street" \
-respath ./workdir/neighborhood \
-d 42
```

### Case: Macaw

**Preview:**

| <img src="./img/Macaw/svg_iter0.svg"> | <img src="./img/Macaw/svg_iter100.svg"> | <img src="./img/Macaw/visual_best.svg"> |
|---------------------------------------|-----------------------------------------|-----------------------------------------|
| Strokes Initialization                | 100 step                                | 1200 step                               |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=1 num_paths=96 sds.warmup=1500 num_iter=2000" \
-pt "macaw full color, ultra detailed, realistic, insanely beautiful" \
-respath ./workdir/macaw \
-d 8091
```

### Case: Bunny

**Preview:**

| <img src="./img/Bunny/svg_iter0.svg"> | <img src="./img/Bunny/svg_iter100.svg"> | <img src="./img/Bunny/visual_best.svg"> |
|---------------------------------------|-----------------------------------------|-----------------------------------------|
| Strokes Initialization                | 100 step                                | 670 step                                |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=5 num_paths=96 softmax_temp=0.5 num_iter=2000" \ 
-pt "portrait of two white bunnies, super realistic, highly detailed" \
-respath ./workdir/bunny \
-d 9001
```

### Case: Dragon

**Preview:**

| <img src="./img/Dragon/svg_iter0.svg"> | <img src="./img/Dragon/svg_iter100.svg"> | <img src="./img/Dragon/visual_best.svg"> |
|----------------------------------------|------------------------------------------|------------------------------------------|
| Strokes Initialization                 | 100 step                                 | 1200 step                                |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=2 num_paths=64 softmax_temp=0.5 mask_object=True" \
-pt "A dragon flying in the sky, full body" \
-respath ./workdir/dragon  \
-d 8023
```

### Case: Unicorn

**Preview:**

| <img src="./img/unicorn/svg_iter0.svg"> | <img src="./img/unicorn/svg_iter100.svg"> | <img src="./img/unicorn/visual_best.svg"> |
|-----------------------------------------|-------------------------------------------|-------------------------------------------|
| Strokes Initialization                  | 100 step                                  | 1200 step                                 |

**Script:**

```shell
python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=2" \
-pt "A unicorn is running on the grassland" \
-respath ./workdir/unicorn \
-d 914678
```

**Preview:**

| <img src="./img/unicorn-2/svg_iter0.svg"> | <img src="./img/unicorn-2/svg_iter100.svg"> | <img src="./img/unicorn-2/visual_best.svg"> |
|-------------------------------------------|---------------------------------------------|---------------------------------------------|
| Strokes Initialization                    | 100 step                                    | 1240 step                                   |

**Script:**

```shell
python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=2 softmax_temp=0.5 mask_object=True" \
-pt "A unicorn is running on the grassland" \
-respath ./workdir/unicorn \
-d 9998
```

### Case: Mushroom

**Preview:**

| <img src="./img/Mushroom/svg_iter0.svg"> | <img src="./img/Mushroom/svg_iter100.svg"> | <img src="./img/Mushroom/final_svg.svg"> |
|------------------------------------------|--------------------------------------------|------------------------------------------|
| Strokes Initialization                   | 100 step                                   | 2000 step                                |

**Script:**

```shell
CUDA_VISIBLE_DEVICES=0 python run_painterly_render.py \
-c diffsketcher.yaml \
-eval_step 10 -save_step 10 \
-update "token_ind=4 num_paths=84 comp_idx=0 attn_coeff=1 softmax_temp=0.4 xdog_intersec=False sds.num_aug=4 sds.grad_scale=2 sds.warmup=0 clip.vis_loss=0 clip.num_aug=4 clip.text_visual_coeff=0 perceptual.coeff=0 opacity_delta=0.2 lr_scheduler=True num_iter=2000" \
-pt "a brightly colored mushroom growing on a logt, minimal 2d line drawing. trending on artstation." \
-respath ./workdir/Mushroom \
-d 621024
```
