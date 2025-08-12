# DCM-VG: Detector Consensus Guided Multi-Query Visual Grounding  
**Yanglin Deng, Jinglin Zhou, Hongyao Chen, Wei Zhang, Yijie Zhou, Tianyang Xu**

> **Paper**: *"DCM-VG: Detector Consensus Guided Multi-Query Visual Grounding via Frozen Multimodal Large Language Model"*  
> [README.md](..%2FMMDRFuse%2FREADME.md)

---

## 🚀 Overview  
![workflow](method.png)  
**DCM-VG** takes a single image and multiple text queries, leverages **Qwen2.5-VL-72B** to generate candidate boxes for each query, and then employs **Grounding DINO** to perform local discriminative selection. The final output is the most accurate bounding box for every query.

---

## 📊 Experimental Results  
Ablation-study visualizations (Table 1):  
<p align="center">
  <img src="ablation.png" width="60%" alt="ablation results"/>
</p>

---

## ⚙️ Environment Setup  
We rely on **Grounding DINO** for the detection stage.  
Please follow the [official Grounding DINO repository](https://github.com/IDEA-Research/GroundingDINO) to create the required virtual environment.

---

## 🏆 Reproduce Competition Results  
1. Edit `config.yaml` to match your local paths and hyper-parameters.  
2. Make the script executable and run:
   ```bash
   chmod +x reproduce.sh
   ./reproduce.sh

## 📬 Contact  
If you have any questions or suggestions, feel free to reach out to the maintainers below:

| Name             | Email                          |
|------------------|--------------------------------|
| Yanglin Deng     | yanglin_deng@163.com           |
| Jinglin Zhou     | 6233114044@stu.jiangnan.edu.cn     |
| Hongyao Chen     | 6233111020@stu.jiangnan.edu.cn |
| Wei Zhang        | phenixnull@gmail.com           |
| Yijie Zhou       | zxj165561@gmail.com            |
| Tianyang Xu      | tianyang.xu@jiangnan.edu.cn    |
| Xiao-Jun Wu      | wu_xiaojun@jiangnan.edu.cn     |
| Josef Kittler    | j.kittler@surrey.ac.uk         |

> 💡 **Tip**: When reporting issues or asking questions, please include a clear subject line (e.g., `[ProjectName] Question about …`) to help us respond faster.






