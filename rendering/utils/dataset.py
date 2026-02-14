import os
import json
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass
from .common import to8b


@dataclass
class BlenderData:
    images: np.ndarray    # (N,H,W,3) float32 [0,1]
    poses: np.ndarray     # (N,4,4)
    H: int
    W: int
    focal: float
    i_train: np.ndarray
    i_val: np.ndarray
    i_test: np.ndarray

def load_blender(data_dir: str, half_res: bool = True, split: str = "train") -> BlenderData:
    """
    Expects:
      data_dir/
        transforms_train.json
        transforms_val.json
        transforms_test.json
        *.png
    """
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(data_dir, f"transforms_{s}.json"), "r") as f:
            metas[s] = json.load(f)

    all_imgs, all_poses = [], []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        for frame in meta["frames"]:
            fp = os.path.join(data_dir, frame["file_path"] + ".png")
            img = np.array(Image.open(fp)).astype(np.float32) / 255.0
            if img.shape[-1] == 4:
                # composite on white
                img = img[..., :3] * img[..., 3:4] + (1.0 - img[..., 3:4])
            imgs.append(img)
            poses.append(np.array(frame["transform_matrix"], dtype=np.float32))
        imgs = np.stack(imgs, axis=0)
        poses = np.stack(poses, axis=0)
        all_imgs.append(imgs)
        all_poses.append(poses)
        counts.append(counts[-1] + imgs.shape[0])

    images = np.concatenate(all_imgs, axis=0)
    poses = np.concatenate(all_poses, axis=0)

    H, W = images.shape[1:3]
    camera_angle_x = float(metas["train"]["camera_angle_x"])
    focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

    if half_res:
        H2, W2 = H // 2, W // 2
        images_half = np.zeros((images.shape[0], H2, W2, 3), dtype=np.float32)
        for i in range(images.shape[0]):
            im = Image.fromarray(to8b(images[i]))
            im = im.resize((W2, H2), resample=Image.BILINEAR)
            images_half[i] = np.array(im).astype(np.float32) / 255.0
        images = images_half
        H, W = H2, W2
        focal = focal / 2.0

    i_train = np.arange(counts[0], counts[1])
    i_val   = np.arange(counts[1], counts[2])
    i_test  = np.arange(counts[2], counts[3])

    return BlenderData(images=images, poses=poses, H=H, W=W, focal=focal,
                      i_train=i_train, i_val=i_val, i_test=i_test)