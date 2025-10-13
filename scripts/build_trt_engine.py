#!/usr/bin/env python3
"""构建 PP-OCRv5 TensorRT 引擎的命令行脚本。

示例：
    python scripts/build_trt_engine.py --models det rec --precision fp16
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnxocr import utils
from onnxocr.trt_utils import build_engine, default_ppocrv5_profiles

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("build_trt_engine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["det", "rec"],
        choices=["det", "rec", "cls"],
        help="需要构建的模型类型",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "fp32"],
        help="引擎精度",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=Path("onnxocr/models/ppocrv5/trt"),
        help="输出 engine 存放目录",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 30,
        help="工作空间大小（字节）",
    )
    parser.add_argument(
        "--det-max-side",
        type=int,
        default=960,
        help="检测模型的最长边限制，用于动态 profile",
    )
    parser.add_argument(
        "--rec-batch",
        type=int,
        default=6,
        help="识别模型批大小（与配置一致即可）",
    )
    parser.add_argument(
        "--rec-max-width",
        type=int,
        default=640,
        help="识别模型最大宽度",
    )
    parser.add_argument(
        "--cls-batch",
        type=int,
        default=6,
        help="分类模型批大小",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略时间戳，强制重建引擎",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    profiles = default_ppocrv5_profiles(
        det_max_side=args.det_max_side,
        rec_batch=args.rec_batch,
        rec_max_width=args.rec_max_width,
        cls_batch=args.cls_batch,
    )

    defaults = utils.infer_args()
    params = defaults.parse_args([])

    onnx_paths = {
        "det": Path(params.det_model_dir),
        "rec": Path(params.rec_model_dir),
        "cls": Path(params.cls_model_dir),
    }

    args.engine_dir.mkdir(parents=True, exist_ok=True)

    for model_type in args.models:
        onnx_path = onnx_paths[model_type]
        engine_path = args.engine_dir / f"{model_type}_{args.precision}.engine"
        logger.info("Building %s -> %s", onnx_path, engine_path)
        build_engine(
            onnx_path,
            engine_path,
            profiles[model_type],
            precision=args.precision,
            workspace_size=args.workspace,
            force_rebuild=args.force,
        )


if __name__ == "__main__":
    main()
