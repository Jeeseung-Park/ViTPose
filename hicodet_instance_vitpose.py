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


checkpoint_path = "models/vitpose-l.pth"
HUMAN_IDX = 49
image_size = np.array([192, 256])
padding = 1.25
transform  = Compose([
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])




def main(args):
    device = torch.device("cuda:0")
    config_path = args.config_path
    checkpoint_path = args.model_checkpoint
    
    cfg = Config.fromfile(config_path)
    model = build_posenet(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location=device)
    model = model.to(device)
    
    args.gt_json_path = os.path.join(args.base_dir, args.gt_json_path)
    args.gt_save_json_path = os.path.join(args.base_dir, args.gt_save_json_path)
    args.image_dir = os.path.join(args.base_dir, args.image_dir)

    with open(args.gt_json_path) as f:
        gt_json = json.load(f)
    
    annos = gt_json['annotation']
    filenames = gt_json["filenames"]

    for anno, filename in tqdm(zip(annos, filenames)):
        image_path = os.path.join(args.image_dir, filename)
        image = im.open(image_path).convert("RGB")
        width, height = image.size  

        bbox_list = anno['boxes_h']

        boxes_o = np.array(anno['boxes_o'])
        object_idx = np.array(anno['object'])
        append_list = boxes_o[object_idx==HUMAN_IDX].tolist()
        #if len(append_list)!=0:
        #    print(filename)
        bbox_list = bbox_list + append_list
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
        

        anno['human_joints'] = human_joints_result
        anno['human_joints_score'] = human_joints_score_result
        

    with open(args.gt_save_json_path, "w") as f:
        json.dump(gt_json, f)

    





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="./", type=str)
    parser.add_argument("--model_checkpoint", default="ViTPose/models/vitpose-l.pth", type=str)
    parser.add_argument("--config_path", default="ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py", type=str)
    
    
    parser.add_argument("--gt_json_path", default="hicodet/instances_test2015.json", type=str)
    parser.add_argument("--gt_save_json_path", default="hicodet/instances_test2015_vitpose.json", type=str)
    parser.add_argument("--image_dir", default="hicodet/hico_20160224_det/images/test2015", type=str)
      
    args = parser.parse_args()
    
    main(args)