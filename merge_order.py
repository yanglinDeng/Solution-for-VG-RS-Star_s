#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

def load_json(p: Path) -> List[Dict[str, Any]]:
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def merge_results(base_path: Path,
                  result_paths: List[Path]) -> List[Dict[str, Any]]:
    """按 result_paths 的顺序（越靠后优先级越高）合并结果"""
    base = load_json(base_path)

    # 倒序读取，保证后出现的覆盖先出现的
    lookup: Dict[tuple, Dict[str, Any]] = {}
    for p in reversed(result_paths):
        for rec in load_json(p):
            key = (rec.get('image_path'), rec.get('question'))
            lookup[key] = rec            # 后写入的优先级高

    # 按 base 的顺序输出
    merged = []
    for sample in base:
        key = (sample.get('image_path'), sample.get('question'))
        if key in lookup:
            merged.append(lookup[key])
    return merged

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple model-result JSON files according to priority."
    )
    parser.add_argument('--base',
                        type=Path,
                        required=True,
                        help='Path to the base JSON (defines the final order)')
    parser.add_argument('--results',
                        type=Path,
                        nargs='+',
                        required=True,
                        help='Result JSON files in priority order (last one wins)')
    parser.add_argument('--output',
                        type=Path,
                        required=True,
                        help='Where to save the merged JSON')

    args = parser.parse_args()

    merged = merge_results(args.base, args.results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f_out:
        json.dump(merged, f_out, ensure_ascii=False, indent=2)

    print(f'Merged {len(merged)} records → {args.output}')

if __name__ == '__main__':
    main()