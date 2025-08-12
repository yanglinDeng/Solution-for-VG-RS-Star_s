#!/usr/bin/env bash
# 用法：./run.sh
# 依赖：python3  + PyYAML（pip install pyyaml）
set -euo pipefail

CFG="config.yaml"
PYTHON=python3

# ---------- 检查 python 与 PyYAML ----------
command -v "$PYTHON" >/dev/null 2>&1 || { echo "请先安装 python3"; exit 1; }
"$PYTHON" -c "import yaml, sys; print('PyYAML OK')" || \
    { echo "请先 pip install pyyaml"; exit 1; }
#
# ---------- 1~4 轮：step1 + step2 ----------
#for idx in 0 1 2 3; do
#    echo "======== 第 $idx 轮 (ind=$idx) ========"
#
#    # ★ 用纯 Python 一行脚本就地改 YAML，不再调用 yq/jq
#    "$PYTHON" - <<'PY' $idx "$CFG"
#import yaml, sys, os
#idx, cfg_path = int(sys.argv[1]), sys.argv[2]
#with open(cfg_path) as f:
#    data = yaml.safe_load(f)
#data['step1']['ind'] = idx
#data['step2']['ind'] = idx
#with open(cfg_path, 'w') as f:
#    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
#PY
#
#    "$PYTHON" qwen72b_prompt_1.py --config "$CFG"
#    "$PYTHON" qwen72b_prompt_2.py --config "$CFG"
#done

## ---------- 2 轮后：step3 + step4 ----------
#echo "======== 开始 step3 ========"
#"$PYTHON" merge_order.py  --config "$CFG"

echo "======== 开始 step4 ========"
"$PYTHON" dino_select.py  --config "$CFG"

echo "全部完成！"

#echo "======== 开始 step1 ========"
#"$PYTHON" qwen72b_prompt_1.py  --config "$CFG"