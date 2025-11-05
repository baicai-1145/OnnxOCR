如果项目对您有帮助，欢迎点击右上角 **Star** 支持！✨  
### **OnnxOCR**  
### ![onnx_logo](onnxocr/test_images/onnxocr_logo.png)  

**基于 ONNX 的高性能多语言 OCR 引擎**  
![GitHub stars](https://img.shields.io/github/stars/jingsongliujing/OnnxOCR?style=social)  
![GitHub forks](https://img.shields.io/github/forks/jingsongliujing/OnnxOCR?style=social)  
![GitHub license](https://img.shields.io/github/license/jingsongliujing/OnnxOCR)  
![Python Version](https://img.shields.io/badge/python-≥3.6-blue.svg)  


## 🚀 版本更新  
- **2025.05.21**  
  1. 新增 PP-OCRv5 模型，单模型支持 5 种文字类型：简体中文、繁体中文、中文拼音、英文和日文。  
  2. 整体识别精度相比ppocrv4提升13个百分点
  3. 精度与Paddleocr3.0保持一致。


## 🌟 核心优势  
1. **脱离深度学习训练框架**：可直接用于部署的通用 OCR。  
2. **跨架构支持**：在算力有限、精度不变的情况下，使用 PaddleOCR 转成 ONNX 模型，重新构建的可部署在 ARM 架构和 x86 架构计算机上的 OCR 模型。  
3. **高性能推理**：在同样性能的计算机上推理速度加速。  
4. **多语言支持**：单模型支持 5 种文字类型：简体中文、繁体中文、中文拼音、英文和日文。  
5. **模型精度**：与 PaddleOCR 模型保持一致。
6. **国产化适配**：重构代码工程架构，只需简单进行推理引擎的修改，即可适配更多国产化显卡。



## 🛠️ 环境安装  
```bash  
python>=3.6  

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt  
```  

**注意**：  
- 默认使用 Mobile 版本模型，使用 PP-OCRv5_Server-ONNX 模型效果更佳。  
- Mobile 模型已存在于 `onnxocr/models/ppocrv5` 下，无需下载；  
- PP-OCRv5_Server-ONNX 模型过大，已上传至 [百度网盘](https://pan.baidu.com/s/1hpENH_SkLDdwXkmlsX0GUQ?pwd=wu8t)（提取码: wu8t），下载后将 `det` 和 `rec` 模型放到 `./models/ppocrv5/` 下替换即可。  

## ⚡ TensorRT 加速

使用建议
- 推荐使用“动态 shape 的 TensorRT”。在我们的测试环境下：
  - 在 RTX 4090 上，相比 ONNX GPU 约 85× 加速；
  - 相比 Ryzen 5 9600X（ONNX CPU）约 15× 加速；
  - 实际效果因数据、驱动与 TensorRT 版本有所差异；为保证精度一致，建议使用 FP32。

当前 OnnxOCR 已内建 TensorRT 推理链路，针对 PP-OCRv5 Server 模型可获得显著速度提升（内部测试约 80×）。

1. **安装 TensorRT 运行时**  
   - 确保 GPU 驱动和 CUDA 版本与 TensorRT 匹配（示例：CUDA 12.9 + TensorRT 10.13）。  
   - 安装官方提供的 Python wheel，例如：  
     ```bash
     uv pip install tensorrt-10.13.3.9-cp311-none-linux_x86_64.whl \
                    tensorrt_dispatch-10.13.3.9-cp311-none-linux_x86_64.whl \
                    tensorrt_lean-10.13.3.9-cp311-none-linux_x86_64.whl
     ```  
   - 如手工解压了 TensorRT 安装包，运行前需要：  
     ```bash
     export LD_LIBRARY_PATH=/path/to/TensorRT/lib:$LD_LIBRARY_PATH
     ```

2. **构建 TensorRT 引擎**  
   使用项目脚本生成检测 / 识别 / 方向分类引擎，并放宽动态 shape（Server 模型建议扩大识别宽度）：  
   ```bash
   python scripts/build_trt_engine.py \
       --models det rec cls \
       --precision fp32 \
       --det-max-side 1280 \
       --rec-max-width 1280 \
       --cls-batch 6 \
       --workspace $((1<<31)) \
       --force
   ```

3. **执行 FP32 推理（推荐）**  
   ```bash
   python test_ocr.py \
       --use-gpu --use-tensorrt \
       --trt-precision fp32 \
       --trt-engine-dir onnxocr/models/ppocrv5/trt \
       --warmup 3 --repeat 10
   ```
   > ⚠️ 支持 `--trt-precision fp16`，但我们在 PP-OCRv5 Server 模型上观察到一定精度损失。如需线上使用请先评估。


## 🚀 一键运行  
```bash  
python test_ocr.py  
```  


## 📡 API 服务（CPU 示例）  
### 启动服务  
```bash  
python app-service.py  
```  

### 测试示例  
#### 请求  
```bash  
curl -X POST http://localhost:5005/ocr \  
-H "Content-Type: application/json" \  
-d '{"image": "base64_encoded_image_data"}'  
```  

#### 响应  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "名称",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "标头",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🐳 Docker 镜像环境（CPU）  
### 镜像构建  
```bash  
docker build -t ocr-service .  
```  

### 镜像启动  
```bash  
docker run -itd --name onnxocr-service-v3 -p 5006:5005 onnxocr-service:v3  
```  

### POST 请求  
```  
url: ip:5006/ocr  
```  

### 返回值示例  
```json  
{  
  "processing_time": 0.456,  
  "results": [  
    {  
      "text": "名称",  
      "confidence": 0.9999361634254456,  
      "bounding_box": [[4.0, 8.0], [31.0, 8.0], [31.0, 24.0], [4.0, 24.0]]  
    },  
    {  
      "text": "标头",  
      "confidence": 0.9998759031295776,  
      "bounding_box": [[233.0, 7.0], [258.0, 7.0], [258.0, 23.0], [233.0, 23.0]]  
    }  
  ]  
}  
```  


## 🌟 效果展示  
| 示例 1 | 示例 2 |  
|--------|--------|  
| ![](result_img/r1.png) | ![](result_img/r2.png) |  

| 示例 3 | 示例 4 |  
|--------|--------|  
| ![](result_img/r3.png) | ![](result_img/draw_ocr4.jpg) |  

| 示例 5 | 示例 6 |  
|--------|--------|  
| ![](result_img/draw_ocr5.jpg) | ![](result_img/555.png) |  


## 👨💻 联系与交流  
### 求职信息  
本人正在寻求工作机会，欢迎联系！  
![微信二维码](onnxocr/test_images/myQR.jpg)  

### OnnxOCR 交流群  
#### 微信群  
![微信群](onnxocr/test_images/微信群.jpg)  

#### QQ 群  
![QQ群](onnxocr/test_images/QQ群.jpg)  




## 🎉 致谢  
感谢 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提供的技术支持！  


## 🌍 开源与捐赠  
我热爱开源和 AI 技术，相信它们能为有需要的人带来便利和帮助，让世界变得更美好。如果您认可本项目，可以通过支付宝或微信进行打赏（备注请注明支持 OnnxOCR）。  

<img src="onnxocr/test_images/weixin_pay.jpg" alt="微信支付" width="200">
<img src="onnxocr/test_images/zhifubao_pay.jpg" alt="支付宝" width="200">


## 📈 Star 历史  
[![Star History Chart](https://api.star-history.com/svg?repos=jingsongliujing/OnnxOCR&type=Date)](https://star-history.com/#jingsongliujing/OnnxOCR&Date)  


## 🤝 贡献指南  
欢迎提交 Issues 和 Pull Requests，共同改进项目！  
