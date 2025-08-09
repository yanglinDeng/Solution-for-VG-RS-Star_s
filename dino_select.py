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

def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Load multiple JSON result files."
    )
    # 接受 1 个或多个 JSON 文件路径
    parser.add_argument(
        "json_files",
        type=Path,
        nargs='+',
        help="List of JSON result files to load"
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to the Grounding DINO model.")
    parser.add_argument("--root_path", type=Path, required=True,
                        help="Path to the root dir of images and questions.json.")
    parser.add_argument("--merged_path", type=Path, required=True,
                        help="Path to the merged comprehensive version.")
    parser.add_argument("--saved_json_path", type=Path, required=True,
                        help="Path to the saved results.")
    args = parser.parse_args()

    # 简单校验：必须都是存在的 .json 文件
    for p in args.json_files:
        if not (p.is_file() and p.suffix.lower() == '.json'):
            parser.error(f"Invalid JSON file: {p}")
    return args

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
def find_box_iou(res,img_path,question,model):
    root = args.root
    box = find_box(res,img_path,question)
    if box is None:
        return 0,0,None
    _,iou,text = run_groundingdino_on_image(img_path,question,box,model,device=0,root_dir=dir)
    return iou,box,text

def main(args):
    model = load_model(args.model_path)
    model = model.cuda()



    with args.merged_path.open('r', encoding='utf-8') as f:
        data_merge = json.load(f)

    all_data = []
    for p in args.json_files:
        with p.open('r', encoding='utf-8') as f:
            all_data.append(json.load(f))
    data0 = all_data[0]
    for i in tqdm(range(len(data0))):
        img_path = data0[i]['image_path']
        ques = data0[i]['question']
        miou,tar_box,tar_tex = find_box_iou(all_data[1], img_path, ques,model)
        if miou==0:
            print(tar_tex)
        for j in range(2,len(all_data)):
            ciou, cbox, ctex = find_box_iou(all_data[j], img_path, ques,model)
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
        with open(args.saved_json_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    args = get_args_parser()
    main(args)


