import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GroundingDINO'))
from dino_utils import run_groundingdino_on_image
from groundingdino.util.inference import load_model
import yaml


def convert_numpy_types(obj):
    """递归地将numpy数据类型转换为Python原生类型以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj


# 找某个版本里的对应框
def find_box(res,img_path,question):
    for t in range(len(res)):
        if res[t]["image_path"]==img_path and res[t]["question"]==question:
            return res[t]['result']
    return None
def find_box_iou(res,img_path,question,model,cfg):
    root = cfg["root_path"]
    box = find_box(res,img_path,question)
    if box is None:
        return 0,0,None
    _,iou,text = run_groundingdino_on_image(img_path,question,box,model,device=0,root_dir=root)
    return iou,box,text

def main(cfg):
    model = load_model(cfg["model_config"],cfg["model_weights"])
    model = model.cuda()

    with open(cfg["merged_path"],'r', encoding='utf-8') as f:
        data_merge = json.load(f)

    all_data = []
    for p in cfg["json_files"]:
        with open(p,'r', encoding='utf-8') as f:
            all_data.append(json.load(f))
    data0 = all_data[0]
    for i in tqdm(range(len(data0))):
        img_path = data0[i]['image_path']
        ques = data0[i]['question']
        miou,tar_box,tar_tex = find_box_iou(all_data[1], img_path, ques,model,cfg)
        if miou==0:
            print(tar_tex)
        for j in range(2,len(all_data)):
            ciou, cbox, ctex = find_box_iou(all_data[j], img_path, ques,model,cfg)
            if max(miou,ciou)!= 0:
                if max(miou,ciou)==ciou:
                    print("已挑选最合适的！")
                    print("选中{}".format(j - 1))
                    tar_box = cbox
                    miou = ciou
                else:
                    print("已挑选最合适的！")
                    print("选中{}".format(1))
            else:
                print(ctex)
        if tar_box == 0:
            tar_box = find_box(data_merge,img_path,ques)
            if tar_box is None:
                continue
        result = {
            "image_path": img_path,
            "question": ques,
            "result": tar_box
        }
        # content_list.append(result)
        with open(cfg["saved_json_path"], "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
        exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='全局配置文件路径')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    main(cfg['step4'])
