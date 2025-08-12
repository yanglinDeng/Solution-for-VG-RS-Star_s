import argparse
import base64
import cv2
from pathlib import Path
import numpy as np
import random
import ast
from PIL import Image
import os,yaml
from openai import OpenAI
import re
import json
result = []
failed_list=[]

def encode_local_image(image_path,cfg):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (cfg["width"][cfg["ind"]],cfg["height"][cfg["ind"]]))
    _, jpeg_bytes = cv2.imencode('.jpg', resized_image)
    base64_data = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_data}"




def get_position_info(image_path, ques_str,cfg):
    try:
        client = OpenAI(
            api_key=cfg["api_key"],
            base_url=cfg["base_url"]
        )

        image_data_url = encode_local_image(image_path,cfg)

        format_instruction = (
            "I will provide you with an image and several questions. "
            "Each question is about the single bounding box of a specific object in the image. "
            "Your task is to analyze the image and the positional relationships between objects, "
            "and then return the accurate 2D bounding box coordinates for all main objects mentioned in the questions. "
            "Please return the results in the following universal JSON format:"
            "{"
            "  'result_1': {'label': 'object_label_1', 'bbox_2d': [x1, y1, x2, y2]},"
            "  'result_2': {'label': 'object_label_2', 'bbox_2d': [x1, y1, x2, y2]},"
            "  ..."
            "  'result_n': {'label': 'object_label_n', 'bbox_2d': [x1, y1, x2, y2]}"
            "}"
            "Explanation of fields:"
            "- 'label': original question"
            "- 'bbox_2d': the 2D bounding box coordinates in format [x1, y1, x2, y2], "
            "  where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box."
            "If an object mentioned in a question is not found in the image, set its 'bbox_2d' value to [0, 0, 0, 0]."
            )
        
        completion = client.chat.completions.create(
            model=cfg["model_name"],

            messages=[
                {"role": "user", "content": [{"type": "text", "text": format_instruction}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            },
                            "detail": "high",
                            "min_pixels":cfg["height"][cfg["ind"]]*cfg["width"][cfg["ind"]]-1,
                            "max_pixels":cfg["height"][cfg["ind"]]*cfg["width"][cfg["ind"]]+1
                        },
                        {"type": "text", "text": ques_str}
                    ]
                }
            ],
        )

        result_data = completion.model_dump()
        return result_data['choices'][0]['message']['content']
    except Exception as e:
        print(f"{str(e)}")
        return None


def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output



def res_error(img_path,ques):
    fail = {
        "image_path": img_path,
        "question": ques
    }
    return fail

def detect(data, img_path, ques_list, ind, jnd, ori_ques_list, oh, ow,cfg):
    image = cv2.imread(img_path)
    answer = get_position_info(img_path, str(ques_list),cfg)
    if answer is None:
        failed_list.append(res_error(img_path, ori_ques_list[0]))
        return ind + 1
    answer_str = parse_json(answer)
    print(answer_str)
    try:
        answer_object = ast.literal_eval(answer_str)
    except Exception as e:
        end_idx = answer_str.rfind('"}') + len('"}')
        truncated_text = answer_str[:end_idx] + "]"
        try:
            answer_object = ast.literal_eval(truncated_text)
        except Exception as e:
            failed_list.append(res_error(img_path, ori_ques_list[0]))
            return ind + 1
    print(answer_object)
    if isinstance(answer_object, dict):
        if (len(list(answer_object.keys()))) != len(ques_list):
            failed_list.append(res_error(img_path, ori_ques_list[0]))
            return ind + 1
        for s in range(len(list(answer_object.keys()))):
            skey = list(answer_object.keys())[s]
            if skey[:6] == "result":
                try:
                    len(answer_object[skey]["bbox_2d"]) != 4
                except KeyError:
                    failed_list.append(res_error(img_path, ori_ques_list[s]))
                    continue
                except TypeError:
                    failed_list.append(res_error(img_path, ori_ques_list[s]))
                    continue
                try:
                    abs_y1 = answer_object[skey]["bbox_2d"][1]
                    abs_x1 = answer_object[skey]["bbox_2d"][0]
                    abs_y2 = answer_object[skey]["bbox_2d"][3]
                    abs_x2 = answer_object[skey]["bbox_2d"][2]
                except IndexError:
                    res_error(img_path, ques_list[s])
                    continue
                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                if abs_x1 == 0 and abs_x2 == 0 and abs_y1 == 0 and abs_y2 == 0:
                    failed_list.append(res_error(img_path, ori_ques_list[s]))
                    continue
                tx1, tx2 = round((ow / cfg["width"][cfg["ind"]]) * abs_x1), round((ow / cfg["width"][cfg["ind"]]) * abs_x2)
                ty1, ty2 = round((oh / cfg["height"][cfg["ind"]]) * abs_y1), round((oh / cfg["height"][cfg["ind"]]) * abs_y2)
                content = {
                    "image_path": data[ind]['image_path'],
                    'question': ori_ques_list[s],
                    "result": [[tx1, ty1], [tx2, ty2]]}
                result.append(content)
                image = cv2.resize(image, (ow, oh))
                cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 255, 0), 1)
                cv2.putText(image, answer_object[skey]['label'], (tx1, ty1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, answer_object[skey]['label'], (tx1, ty1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(cfg["success_vis_dir"][cfg["ind"]] +"/"+ data[ind]["image_path"].split("images\\")[1], image)
    with open(cfg["success_json"][cfg["ind"]], "w+", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("success for the {} samples!".format(jnd))
    return jnd


def main(cfg):
    # ------------------------------------------hongyao--------------------------------------------------------------------------------
    seed = cfg["seed"]
    print(f"Using random seed: {seed}")
    np.random.seed(seed)
    random.seed(seed)
    root = cfg["image_path"]
    json_path = cfg["question_path"]
    with open(json_path,'r', encoding='utf-8') as f:
        data = json.load(f)
    success_vis_path = Path(cfg["success_vis_dir"][cfg["ind"]])  # 换成你的路径

    success_vis_path.mkdir(parents=True, exist_ok=True)
    # --------------------------------test for all samples------------------------
    i = 0
    while i < len(data):
        path = data[i]["image_path"].split("images\\")[1]
        img_path = os.path.join(root, path)
        ori_ques_list = []
        ori_ques_list.append(data[i]["question"])
        j = i + 1
        while j < len(data) and data[j]["image_path"] == data[i]["image_path"]:
            ori_ques_list.append(data[j]["question"])
            j += 1
        print(ori_ques_list)
        with Image.open(img_path) as img:
            ow, oh = img.size  # (W, H)
        print(ow, oh)
        ques_list = [q.replace('\n', '').replace('\t', '') for q in ori_ques_list]
        i = detect(data, img_path, ques_list, i, j, ori_ques_list, oh, ow,cfg)

    with open(cfg["failure_json"][cfg["ind"]], "w+", encoding="utf-8") as f:
        json.dump(failed_list, f, ensure_ascii=False, indent=2)
    print("success for the {} failed samples!".format(len(failed_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='全局配置文件路径')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    main(cfg['step1'])
