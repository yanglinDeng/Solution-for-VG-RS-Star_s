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
You should first edit the "config.yaml" for your own settings.
After you have done all the preparation work, you can run following commands:

```
chmod +x run.sh
./run.sh
```


## Contact Informaiton
If you have any questions, please contact me at <yanglin_deng@163.com>
