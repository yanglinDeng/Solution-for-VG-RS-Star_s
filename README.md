# DCM-VG
Yanglin Deng, Jinglin Zhou, Hongyao Chen, Wei Zhang, Yijie Zhou, Tianyang Xu
"DCM-VG: Detector Consensus Guided Multi-Query Visual Grounding via
Frozen Multimodal Large Language Model[README.md](..%2FMMDRFuse%2FREADME.md)".

---
![Abstract](method.png)
---
The workflow of our proposed DCM-VG. A single image with multiple questions is grounded through Qwen2.5-VL-72B for
multiple queries. Hand over candidate boxes from multiple results to GroundingDINO for local discriminative selection, and finally
select the most accurate bounding box.

## Experimental results display
Visualization results of ablation study in Table 1:
<img src="ablation.png" width="60%" align=center />

## Virtual Environment
Please refer to the official Grounding DINO repository for the complete environment setup.

## To reproduce the competition results.
You should first use API of model Qwen2.5-VL-72B-Instruct to produce several grounding results, 
which are prmopted by prompt1.
In our work, we use four resolutions: 1120×1120, 2240×3360, 3360×2240, 3360×3360.
Please run with following steps:
```
python qwen72b_prompt_1.py \
  --width 1120 \
  --height 1120 \
  --model_name Qwen2.5-VL-72B-Instruct \
  --api_key sk-xxxxxxxxxxxxxxxx \
  --base_url https:xxxxxxxxxxxxxxxx \
  --image_path      "xxxxxx/Solution of VG-RS/VG-RS/images" \
  --question_path   "xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json" \
  --failure_json    "xxxxxx/Solution of VG-RS/failed_1120_1120.json" \
  --success_json    "xxxxxx/Solution of VG-RS/success_1120_1120.json" \
  --success_vis_dir "xxxxxx/Solution of VG-RS/success_vis_1120_1120"
```

Then, you can further conduct fine-grained grounding for the remained failure samples prompted by prompt2.
Please run with following steps:
```
python run.py \
  --width 1120 \
  --height 1120 \
  --model_name Qwen2.5-VL-72B-Instruct \
  --api_key sk-xxxxxxxxxxxxxxxx \
  --base_url https:xxxxxxxxxxxxxxxx \
  --image_path      "xxxxxx/Solution of VG-RS/VG-RS/images" \
  --question_path   "xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json" \
  --recored_failed_samples_json " xxxxxx/Solution of VG-RS/failed_1120_1120.json \"
  --failure_json    "xxxxxx/Solution of VG-RS/second_failed_1120_1120.json" \
  --success_json    "xxxxxx/Solution of VG-RS/second_success_1120_1120.json" \
  --success_vis_dir "xxxxxx/Solution of VG-RS/second_success_vis_1120_1120"
```
Then, you should combine the two stages results into a comprehensive version.
You can also use the "merge_order.py" to combine all the comprehensive version of above four resolutions, which is the way to obtain the results in the fifth row of Table 1.
```
python merge_order.py \
  --base  xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json \
  --results \
      xxxxxx/Solution of VG-RS/success_1120_1120.json \
      xxxxxx/Solution of VG-RS/second_success_1120_1120.json \
  --output xxxxxx/Solution of VG-RS/merged_1120_1120.json
  
python merge_order.py \
  --base  xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json \
  --results \
      xxxxxx/Solution of VG-RS/merged_1120_1120.json \
      xxxxxx/Solution of VG-RS/merged_2240_3360.json \
      xxxxxx/Solution of VG-RS/merged_3360_2240.json \
      xxxxxx/Solution of VG-RS/merged_3360_3360.json \
  --output xxxxxx/Solution of VG-RS/merged_total.json
```
Finally, you can exploit the GroundingDINO to conduct selection among above four results(1120×1120, 2240×3360, 3360×2240, 3360×3360).
```
python dino_select.py \
  xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json \
  xxxxxx/Solution of VG-RS/merged_1120_1120.json \
  xxxxxx/Solution of VG-RS/merged_3360_2240.json \
  xxxxxx/Solution of VG-RS/merged_2240_3360.json \
  xxxxxx/Solution of VG-RS/merged_3360_3360.json \
  --model_path  xxxxxx/Solution of VG-RS/GroundingDINO/weights/groundingdino_swint_ogc.pth \
  --root_path   xxxxxx/Solution of VG-RS \
  --merged_path   xxxxxx/Solution of VG-RS/merged_total.json
  --saved_json_path xxxxxx/Solution of VG-RS/selected_results.json
```

You can according to the following code to filter out some failure samples:
```
python filter_failedres.py \
    --base   xxxxxx/Solution of VG-RS/VG-RS/VG-RS-question.json \
    --result xxxxxx/Solution of VG-RS/res_1.json \
             xxxxxx/Solution of VG-RS/res_2.json \
    --output res_failed.json
```

## Contact Informaiton
If you have any questions, please contact me at <yanglin_deng@163.com>




