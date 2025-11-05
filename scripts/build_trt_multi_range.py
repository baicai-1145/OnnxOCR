from __future__ import annotations

"""
Build FP32 TensorRT engines with multi-range dynamic profiles:
- det: single dynamic profile (range tightened)
- rec: 3 dynamic profiles for width ranges (h=48)
- cls: 3 dynamic profiles for width ranges (h=48)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnxocr import utils  # noqa: E402
from onnxocr.trt_utils import (  # noqa: E402
    ProfileShape,
    build_engine,
    build_multi,
    default_ppocrv5_profiles,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("build_trt_multi_range")


def _parse_ranges(text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for seg in (text or "").split(","):
        seg = seg.strip()
        if not seg:
            continue
        lo_s, hi_s = seg.split(":")
        lo, hi = int(lo_s), int(hi_s)
        if lo >= hi:
            raise ValueError(f"Invalid range: {seg}")
        ranges.append((lo, hi))
    if not ranges:
        raise ValueError("Empty ranges")
    return ranges


def _make_profiles_for_ranges(
    ranges: List[Tuple[int, int]],
    *,
    batch_max: int,
    height: int,
    channels: int = 3,
    input_name: str = "x",
) -> List[dict]:
    """
    为多段动态范围生成 profile 列表（min/opt/max）。
    opt 宽取区间中点或略偏上，以适配常见宽度。
    """
    profiles: List[dict] = []
    for lo, hi in ranges:
        opt_w = (lo + hi) // 2
        profiles.append(
            {
                input_name: ProfileShape(
                    min=(1, channels, height, lo),
                    opt=(batch_max, channels, height, opt_w),
                    max=(batch_max, channels, height, hi),
                )
            }
        )
    return profiles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", default="fp32", choices=["fp32"], help="precision (fp32 only)")
    p.add_argument("--engine-dir", type=Path, default=Path("onnxocr/models/ppocrv5/trt_mrange"))
    p.add_argument("--workspace", type=int, default=1 << 30)
    p.add_argument(
        "--models",
        nargs="+",
        choices=["det", "rec", "cls"],
        default=["det", "rec", "cls"],
        help="选择要构建的模型子集（默认全部）",
    )
    p.add_argument("--det-max-side", type=int, default=960)
    p.add_argument("--rec-ranges", type=str, default="32:144,144:480,480:2048")
    p.add_argument("--cls-ranges", type=str, default="24:80,80:144,144:192")
    p.add_argument("--rec-batch", type=int, default=6)
    p.add_argument("--cls-batch", type=int, default=6)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.engine_dir.mkdir(parents=True, exist_ok=True)

    # 模型路径默认从 utils.infer_args 获取
    defaults = utils.infer_args()
    params = defaults.parse_args([])
    onnx_paths = {
        "det": Path(params.det_model_dir),
        "rec": Path(params.rec_model_dir),
        "cls": Path(params.cls_model_dir),
    }

    # det: 单一动态 profile（范围收紧）
    if "det" in args.models:
        det_engine = args.engine_dir / f"det_{args.precision}.engine"
        det_profiles = default_ppocrv5_profiles(
            det_max_side=args.det_max_side,
            rec_batch=args.rec_batch,
            rec_max_width=640,
            cls_batch=args.cls_batch,
        )["det"]
        logger.info("Building det(dynamic) %s -> %s", onnx_paths["det"], det_engine)
        build_engine(
            onnx_paths["det"],
            det_engine,
            det_profiles,
            precision=args.precision,
            workspace_size=args.workspace,
            force_rebuild=args.force,
        )

    # rec/cls: 三段动态 profile
    rec_ranges = _parse_ranges(args.rec_ranges)
    cls_ranges = _parse_ranges(args.cls_ranges)

    def _mapping():
        items = []
        if "rec" in args.models:
            items.append(("rec", onnx_paths["rec"], args.engine_dir / f"rec_{args.precision}.engine"))
        if "cls" in args.models:
            items.append(("cls", onnx_paths["cls"], args.engine_dir / f"cls_{args.precision}.engine"))
        return items

    profiles_dict = {
        "rec": _make_profiles_for_ranges(rec_ranges, batch_max=args.rec_batch, height=48) if "rec" in args.models else [],
        "cls": _make_profiles_for_ranges(cls_ranges, batch_max=args.cls_batch, height=48) if "cls" in args.models else [],
    }

    # 逐个构建多profile引擎
    for key, onnx_path, engine_path in _mapping():
        logger.info("Building %s multi-range -> %s", key, engine_path)
        from onnxocr.trt_utils import build_engine_multi_profiles

        build_engine_multi_profiles(
            onnx_path,
            engine_path,
            profiles_list=profiles_dict[key],
            precision=args.precision,
            workspace_size=args.workspace,
            force_rebuild=args.force,
        )


if __name__ == "__main__":
    main()
