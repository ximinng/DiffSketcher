from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from skimage.transform import resize

from .u2net import U2NET


def get_mask_u2net(pil_im, output_dir, u2net_path, device="cpu"):
    # input preprocess
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(device)

    # load U^2 Net model
    net = U2NET(in_ch=3, out_ch=1)
    net.load_state_dict(torch.load(u2net_path))
    net.to(device)
    net.eval()

    # get mask
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], dim=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    # predict_np = predict.clone().cpu().data.numpy()
    im = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).convert('RGB')
    save_path_ = output_dir / "mask.png"
    im.save(save_path_)

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)

    # free u2net
    del net
    torch.cuda.empty_cache()

    return im_final, predict
