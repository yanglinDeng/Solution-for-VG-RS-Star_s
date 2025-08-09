#!/usr/bin/env python3
"""
filter_failed.py
Filter samples in --base that do NOT appear in any --result files,
and save them to --output.
Usage:
    python filter_failedres.py \
    --base   /data_C/yanglin/Qwen2.5-VL/VG-RS/VG-RS-question.json \
    --result /data_C/yanglin/Qwen2.5-VL/res.json \
             /data_C/yanglin/Qwen2.5-VL/res_1.json \
    --output res_failed.json
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def load(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def make_key(rec: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Any, ...]:
    return tuple(rec.get(k) for k in keys)

def filter_failed(base_path: Path,
                  result_paths: List[Path],
                  keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """返回 base 中未在任何 result 里出现的记录"""
    # 把所有结果中出现过的 key 放进集合
    found_keys = set()
    for p in result_paths:
        for rec in load(p):
            found_keys.add(make_key(rec, keys))

    # 遍历 base
    base = load(base_path)
    return [rec for rec in base if make_key(rec, keys) not in found_keys]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter samples in base JSON that are absent in any result JSON."
    )
    parser.add_argument('--base', type=Path, required=True,
                        help='Path to the base JSON file')
    parser.add_argument('--result', type=Path, nargs='+', required=True,
                        help='One or more result JSON files')
    parser.add_argument('--output', type=Path, required=True,
                        help='Where to save the filtered failed JSON')
    parser.add_argument('--keys', type=str, nargs='+', default=['image_path', 'question'],
                        help='Fields used as matching key (default: image_path question)')
    return parser.parse_args()

def main():
    args = parse_args()
    failed = filter_failed(args.base, args.result, tuple(args.keys))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f_out:
        json.dump(failed, f_out, ensure_ascii=False, indent=2)

    print(f'Finished: {len(failed)} failed samples → {args.output}')

if __name__ == '__main__':
    main()