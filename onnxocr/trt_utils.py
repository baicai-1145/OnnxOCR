"""TensorRT 构建与加载辅助工具。

该模块封装了从 ONNX 构建 TensorRT Engine 以及加载 Engine 的通用逻辑，
方便检测、识别等不同模型共享，保持 PredictBase 简洁（KISS/DRY）。
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _to_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _cuda_check(status, msg: str) -> None:
    if isinstance(status, tuple):
        status = status[0]
    if status != cudart.cudaError_t.cudaSuccess:
        err_name = _to_string(cudart.cudaGetErrorName(status)[1])
        err_desc = _to_string(cudart.cudaGetErrorString(status)[1])
        raise RuntimeError(f"{msg}: {err_name} ({err_desc})")


@dataclasses.dataclass(frozen=True)
class ProfileShape:
    """描述单个输入张量的动态 shape 范围。"""

    min: Tuple[int, ...]
    opt: Tuple[int, ...]
    max: Tuple[int, ...]

    def as_trt(self) -> Tuple[trt.Dims, trt.Dims, trt.Dims]:
        return (trt.Dims(self.min), trt.Dims(self.opt), trt.Dims(self.max))


ProfileDict = Dict[str, ProfileShape]


def _parse_onnx(network: trt.INetworkDefinition, onnx_path: Path) -> None:
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with onnx_path.open("rb") as model_stream:
        model_bytes = model_stream.read()
    if not parser.parse(model_bytes):
        errors = []
        for i in range(parser.num_errors):
            errors.append(parser.get_error(i))
        error_text = "\n".join(str(err) for err in errors)
        raise RuntimeError(f"Failed to parse ONNX model {onnx_path}:\n{error_text}")


def _build_engine_internal(
    onnx_path: Path,
    profiles: ProfileDict,
    *,
    precision: str = "fp16",
    workspace_size: int = 1 << 30,
    sparse_weights: bool = False,
) -> trt.IHostMemory:
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=network_flags)

    _parse_onnx(network, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logger.warning("FP16 not supported on this platform, fallback to FP32")
    elif precision.lower() == "fp32":
        pass
    else:
        raise ValueError("Only fp16/fp32 precision is currently supported")

    if sparse_weights:
        if builder.platform_has_fast_sparse:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        else:
            logger.warning("Sparse weights not supported, skip")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        if inp.name not in profiles:
            raise KeyError(f"Missing dynamic profile for input {inp.name}")
        shape = profiles[inp.name]
        profile.set_shape(inp.name, shape.min, shape.opt, shape.max)
    config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"Failed to build TensorRT engine from {onnx_path}")
    return engine_bytes


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: ProfileDict,
    *,
    precision: str = "fp16",
    workspace_size: int = 1 << 30,
    sparse_weights: bool = False,
    force_rebuild: bool = False,
) -> Path:
    """从 ONNX 构建 TensorRT 引擎并保存。

    Args:
        onnx_path: ONNX 模型路径。
        engine_path: 输出 engine 路径；父目录自动创建。
        profiles: 输入张量的动态 shape 配置。
        precision: "fp16" 或 "fp32"。
        workspace_size: TensorRT 工作空间大小（字节）。
        sparse_weights: 是否启用稀疏权重。
        force_rebuild: 为 True 时无条件重建；否则仅当 engine 不存在或旧于 onnx。
    Returns:
        engine_path
    """

    onnx_path = onnx_path.resolve()
    engine_path = engine_path.resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if (
        not force_rebuild
        and engine_path.exists()
        and engine_path.stat().st_mtime >= onnx_path.stat().st_mtime
    ):
        logger.info("Skip rebuild, engine is up-to-date: %s", engine_path)
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    engine_bytes = _build_engine_internal(
        onnx_path,
        profiles,
        precision=precision,
        workspace_size=workspace_size,
        sparse_weights=sparse_weights,
    )

    with engine_path.open("wb") as f:
        f.write(engine_bytes)

    logger.info("TensorRT engine saved to %s", engine_path)
    return engine_path


def load_engine(engine_path: Path) -> trt.ICudaEngine:
    """反序列化 TensorRT 引擎。"""

    engine_path = engine_path.resolve()
    if not engine_path.exists():
        raise FileNotFoundError(engine_path)

    runtime = trt.Runtime(TRT_LOGGER)
    with engine_path.open("rb") as f:
        engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
    return engine


class TensorRTSession:
    """提供与 ONNXRuntime Session 兼容接口的 TensorRT 封装。"""

    class _BindingInfo:
        def __init__(self, name: str):
            self.name = name

    def __init__(self, engine: trt.ICudaEngine, device_id: int = 0):
        self.engine = engine
        self.device_id = device_id

        _cuda_check(cudart.cudaSetDevice(device_id), "cudaSetDevice")

        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context for TensorRT engine")

        status, stream = cudart.cudaStreamCreate()
        _cuda_check(status, "cudaStreamCreate")
        self.stream = stream

        if engine.num_optimization_profiles > 0:
            self.context.set_optimization_profile_async(0, self.stream)

        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        logger.debug("[TRT] tensors: %s", tensor_names)
        self._input_names = [
            self._BindingInfo(name)
            for name in tensor_names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self._output_names = [
            self._BindingInfo(name)
            for name in tensor_names
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]

    # 与 onnxruntime.InferenceSession 接口保持一致
    def get_inputs(self) -> List["TensorRTSession._BindingInfo"]:
        return self._input_names

    def get_outputs(self) -> List["TensorRTSession._BindingInfo"]:
        return self._output_names

    def __del__(self):
        try:
            if hasattr(self, "stream"):
                cudart.cudaStreamDestroy(self.stream)
        except Exception:  # pragma: no cover - 清理失败忽略
            pass

    def run(self, output_names: List[str], input_feed: Dict[str, np.ndarray]):
        if not output_names:
            output_names = [info.name for info in self._output_names]

        if self.engine.num_optimization_profiles > 0:
            self.context.set_optimization_profile_async(0, self.stream)

        staged_inputs: List[Tuple[str, np.ndarray]] = []
        for name, array in input_feed.items():
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                raise KeyError(f"Tensor {name} is not registered as input")

            expected_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host_array = np.asarray(array, dtype=expected_dtype)
            host_array = np.ascontiguousarray(host_array)

            self.context.set_input_shape(name, host_array.shape)
            staged_inputs.append((name, host_array))
            logger.debug(
                "[TRT] input %s shape=%s dtype=%s",
                name,
                host_array.shape,
                host_array.dtype,
            )

        input_allocations: List[int] = []
        for name, host_array in staged_inputs:
            bytes_size = host_array.nbytes
            status, device_ptr = cudart.cudaMalloc(bytes_size)
            _cuda_check(status, f"cudaMalloc input {name}")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    device_ptr,
                    host_array.ctypes.data,
                    bytes_size,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                ),
                f"cudaMemcpyAsync H2D {name}",
            )
            self.context.set_tensor_address(name, device_ptr)
            input_allocations.append(device_ptr)
            logger.debug(
                "[TRT] set_tensor_address input %s ptr=%s bytes=%d",
                name,
                hex(device_ptr),
                bytes_size,
            )

        outputs_info: List[Tuple[str, np.ndarray, int]] = []
        for name in output_names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                raise KeyError(f"Tensor {name} is not registered as output")
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            host_array = np.empty(shape, dtype=dtype)
            status, device_ptr = cudart.cudaMalloc(host_array.nbytes)
            _cuda_check(status, f"cudaMalloc output {name}")
            self.context.set_tensor_address(name, device_ptr)
            outputs_info.append((name, host_array, device_ptr))
            logger.debug(
                "[TRT] output %s shape=%s dtype=%s ptr=%s",
                name,
                shape,
                dtype,
                hex(device_ptr),
            )

        success = self.context.execute_async_v3(stream_handle=self.stream)
        if not success:
            raise RuntimeError("TensorRT execution failed")

        for name, host_array, device_ptr in outputs_info:
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    host_array.ctypes.data,
                    device_ptr,
                    host_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                ),
                f"cudaMemcpyAsync D2H {name}",
            )

        _cuda_check(cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

        for device_ptr in input_allocations:
            _cuda_check(cudart.cudaFree(device_ptr), "cudaFree input")
        for _, _, device_ptr in outputs_info:
            _cuda_check(cudart.cudaFree(device_ptr), "cudaFree output")

        return [host_array for _, host_array, _ in outputs_info]



def default_ppocrv5_profiles(
    *,
    det_max_side: int = 960,
    rec_batch: int = 6,
    rec_max_width: int = 640,
    cls_batch: int = 6,
) -> Dict[str, ProfileDict]:
    """返回 PP-OCRv5 常用的动态 profile。"""

    det_profile = {
        "x": ProfileShape(
            min=(1, 3, 320, 320),
            opt=(1, 3, min(det_max_side, 640), min(det_max_side, 640)),
            max=(1, 3, det_max_side, det_max_side),
        )
    }

    rec_profile = {
        "x": ProfileShape(
            min=(1, 3, 48, 32),
            opt=(rec_batch, 3, 48, min(rec_max_width, 320)),
            max=(rec_batch, 3, 48, rec_max_width),
        )
    }

    cls_profile = {
        "x": ProfileShape(
            min=(1, 3, 32, 32),
            opt=(cls_batch, 3, 48, 192),
            max=(cls_batch, 3, 64, 256),
        )
    }

    return {"det": det_profile, "rec": rec_profile, "cls": cls_profile}


def build_multi(
    mapping: Iterable[Tuple[str, Path, Path]],
    profiles: Dict[str, ProfileDict],
    **kwargs,
) -> None:
    """批量构建多个模型的 TensorRT 引擎。"""

    for key, onnx_path, engine_path in mapping:
        if key not in profiles:
            raise KeyError(f"Missing profile for {key}")
        logger.info("Building %s engine", key)
        build_engine(onnx_path, engine_path, profiles[key], **kwargs)
