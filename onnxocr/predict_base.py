import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import onnxruntime


class PredictBase(object):
    def __init__(self, args=None):
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

    # --- session helpers -------------------------------------------------
    def _create_onnx_session(self, model_path: str, use_gpu: bool):
        providers: List[Any]
        if use_gpu:
            providers = [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        return onnxruntime.InferenceSession(model_path, None, providers=providers)

    def _create_trt_session(self, model_path: str, model_key: str):
        if self.args is None:
            raise RuntimeError("TensorRT session requires runtime args")

        from . import trt_utils
        import re

        engine_dir = getattr(self.args, "trt_engine_dir", None)
        precision = getattr(self.args, "trt_precision", "fp16")
        fallback = getattr(self.args, "trt_fallback_onnx", True)

        if engine_dir:
            engine_dir_path = Path(engine_dir)
            engine_path = engine_dir_path / f"{model_key}_{precision}.engine"
        else:
            engine_path = Path(model_path).with_suffix(".engine")

        try:
            engine = trt_utils.load_engine(engine_path)
        except FileNotFoundError:
            if fallback:
                self.logger.warning(
                    "TensorRT engine %s not found, fallback to ONNXRuntime", engine_path
                )
                return None
            raise

        self.logger.info("Loaded TensorRT engine: %s", engine_path)
        # 如果是多profile且为 rec/cls，则创建范围型多profile会话；否则默认单会话
        if model_key in ("rec", "cls") and engine.num_optimization_profiles > 1:
            def _parse_ranges(text: str) -> list[tuple[int, int]]:
                arr = []
                for seg in re.split(r"[ ,]+", text or ""):
                    if not seg:
                        continue
                    lo, hi = seg.split(":")
                    arr.append((int(lo), int(hi)))
                return arr

            if model_key == "rec":
                ranges = _parse_ranges(getattr(self.args, "rec_ranges", "32:144,144:480,480:1280"))
                max_b = int(getattr(self.args, "rec_batch_num", 6))
            else:
                ranges = _parse_ranges(getattr(self.args, "cls_ranges", "24:80,80:144,144:192"))
                max_b = int(getattr(self.args, "cls_batch_num", 6))
            profile_specs = [(lo, hi, max_b) for (lo, hi) in ranges]
            return trt_utils.TensorRTMultiProfileSession(engine, profile_specs=profile_specs)
        return trt_utils.TensorRTSession(engine)

    def get_session(self, model_path: str, use_gpu: bool, model_key: str):
        use_trt = bool(getattr(self.args, "use_tensorrt", False))
        if use_trt:
            session = self._create_trt_session(model_path, model_key)
            if session is not None:
                return session
        return self._create_onnx_session(model_path, use_gpu)

    # --- utility helpers -------------------------------------------------
    def get_output_name(self, session: Any):
        output_name = []
        if hasattr(session, "get_outputs"):
            nodes = session.get_outputs()
            for node in nodes:
                output_name.append(node.name)
        else:
            raise AttributeError("session object missing get_outputs()")
        return output_name

    def get_input_name(self, session: Any):
        input_name = []
        if hasattr(session, "get_inputs"):
            nodes = session.get_inputs()
            for node in nodes:
                input_name.append(node.name)
        else:
            raise AttributeError("session object missing get_inputs()")
        return input_name

    def get_input_feed(self, input_name: Iterable[str], image_numpy):
        input_feed: Dict[str, Any] = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
