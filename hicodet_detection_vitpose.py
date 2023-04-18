import numpy as np
import torch
from mmpose.models import build_posenet
from mmcv import Config
from mmcv.runner import load_checkpoint
import json
import PIL.Image as im
import os
from mmpose.core.post_processing import get_warp_matrix
import cv2
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
import argparse

device = torch.device("cuda:0")
HUMAN_IDX = 49
image_size = np.array([192, 256])
padding = 1.25
transform  = Compose([
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])




def main(args):
    config_path = args.config_path
    checkpoint_path = args.model_checkpoint
    args.det_save_json_dir = os.path.join(args.base_dir, args.det_save_json_dir)
    args.det_json_dir = os.path.join(args.base_dir, args.det_json_dir)
    args.image_dir = os.path.join(args.base_dir, args.image_dir)
    os.makedirs(args.det_save_json_dir, exist_ok=True)
    
    cfg = Config.fromfile(config_path)
    model = build_posenet(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model = model.to(device)
    det_json_path_list = [os.path.join(args.det_json_dir, name) for name in os.listdir(args.det_json_dir)]
    #image_path_list = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]

    for det_json_path in tqdm(det_json_path_list):
        det_filename = det_json_path.split("/")[-1]
        image_filename = det_filename.replace("json", "jpg")
        image_path = os.path.join(args.image_dir, image_filename)

        image = im.open(image_path).convert("RGB")
        
        with open(det_json_path) as f:
            det_json = json.load(f)

        width, height = image.size
        det_json_labels = np.array(det_json['labels'])
        det_json_boxes = np.array(det_json['boxes'])
        
        human_idx = (det_json_labels == HUMAN_IDX)

        bbox_list = det_json_boxes[human_idx]

        human_joints_result = []
        human_joints_score_result = []
        for bbox in bbox_list:

            x,y = bbox[0], bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            x,y,w,h = x1, y1, x2 - x1, y2 - y1

            aspect_ratio = image_size[0] / image_size[1]
            center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
            # padding to include proper amount of context
            scale = scale * padding

            trans = get_warp_matrix(0, center * 2.0, image_size - 1.0, scale * 200.0)
            processed_img = cv2.warpAffine(
                                np.array(image),
                                trans, (int(image_size[0]), int(image_size[1])),
                                flags=cv2.INTER_LINEAR)
            

            input_tensor = transform(processed_img).to(device)
            img_metas = [{'image_file': None, 'center': center, 'scale': scale, 'rotation': 0, 'bbox_score': 1}]

            result = model.forward(input_tensor.unsqueeze(0), img_metas=img_metas, return_loss=False, return_heatmap=True)
            
            heatmap = result['output_heatmap'][0]
            preds = result['preds'][0]
            boxes = result['boxes'][0]
            
            human_joints = preds[:, :2]
            human_joints_score = np.max(heatmap, axis=(1,2))

            human_joints_result.append(human_joints.tolist())
            human_joints_score_result.append(human_joints_score.tolist())
        
        det_json['human_joints'] = human_joints_result
        det_json['human_joints_score'] = human_joints_score_result

        save_json_path = os.path.join(args.det_save_json_dir, det_filename)
        with open(save_json_path, "w") as f:
            json.dump(det_json, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="./", type=str)
    parser.add_argument("--model_checkpoint", default="ViTPose/models/vitpose-l.pth", type=str)
    parser.add_argument("--config_path", default="ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py", type=str)
    
    parser.add_argument('--image_dir', default="hicodet/hico_20160224_det/images/test2015", type=str)
    parser.add_argument('--det_json_dir', default="hicodet/detections/test2015_gt",type=str)
    parser.add_argument('--det_save_json_dir', default="hicodet/detections/test2015_gt_vitpose", type=str)
    
    
    args = parser.parse_args()
    
    main(args)