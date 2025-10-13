import argparse
import time
from pathlib import Path

import cv2

from onnxocr.onnx_paddleocr import ONNXPaddleOcr, sav2Img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick benchmark for OnnxOCR models.")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("onnxocr/test_images/715873facf064583b44ef28295126fa7.jpg"),
        help="测试图片路径",
    )
    parser.add_argument(
        "--use-gpu",
        dest="use_gpu",
        action="store_true",
        default=True,
        help="启用 GPU 推理 (默认开启)",
    )
    parser.add_argument(
        "--cpu",
        dest="use_gpu",
        action="store_false",
        help="强制使用 CPU 推理",
    )
    parser.add_argument(
        "--use-tensorrt",
        action="store_true",
        help="切换到 TensorRT 引擎",
    )
    parser.add_argument(
        "--trt-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TensorRT 引擎精度 (默认 fp16)",
    )
    parser.add_argument(
        "--trt-engine-dir",
        type=Path,
        default=Path("onnxocr/models/ppocrv5/trt"),
        help="TensorRT 引擎目录",
    )
    parser.add_argument(
        "--no-angle-cls",
        dest="use_angle_cls",
        action="store_false",
        help="禁用方向分类器",
    )
    parser.add_argument(
        "--save-result",
        dest="save_result",
        action="store_true",
        help="保存可视化结果",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="预热次数",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="统计平均耗时的循环次数",
    )
    parser.set_defaults(use_angle_cls=True, save_result=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    model_kwargs = {
        "use_angle_cls": args.use_angle_cls,
        "use_gpu": args.use_gpu,
    }

    if args.use_tensorrt:
        model_kwargs.update(
            {
                "use_tensorrt": True,
                "trt_precision": args.trt_precision,
                "trt_engine_dir": str(args.trt_engine_dir),
                "trt_fallback_onnx": False,
            }
        )

    print("Model options:", model_kwargs)
    model = ONNXPaddleOcr(**model_kwargs)

    img = cv2.imread(str(args.image))
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    # warmup
    for _ in range(max(args.warmup, 0)):
        model.ocr(img)

    timings = []
    repeat = max(args.repeat, 1)
    for _ in range(repeat):
        start = time.perf_counter()
        result = model.ocr(img)
        end = time.perf_counter()
        timings.append(end - start)

    avg = sum(timings) / len(timings)
    print(f"Runs: {repeat}, avg: {avg:.4f}s, min: {min(timings):.4f}s, max: {max(timings):.4f}s")

    if result is None or not result:
        print("No result returned.")
        return

    print("OCR result:")
    for box, rec in result[0]:
        print(box, rec)

    if args.save_result:
        output_name = f"trt_test_{time.strftime('%Y%m%d_%H%M%S')}".replace(".", "_")
        sav2Img(img, result, name=f"{output_name}.jpg")
        print("Saved visualization to", output_name + ".jpg")


if __name__ == "__main__":
    main()
