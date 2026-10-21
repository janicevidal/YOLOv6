# Export ONNX Model

## Check requirements
```shell
pip install onnx>=1.10.0
```

## Export script
```shell
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --simplify
```



#### Description of all arguments

- `--weights` : The path of yolov6 model weights.
- `--output` : Output ONNX file path; defaults to a file beside the checkpoint.
- `--img` : Image size of model inputs.
- `--batch` : Batch size of model inputs.
- `--dynamic-batch` : Keep the batch dimension dynamic; spatial dimensions stay fixed.
- `--opset` : ONNX opset; defaults to 16 when the model contains DySample and 13 otherwise.
- `--half` : Whether to export half-precision model.
- `--inplace` : Whether to set Detect() inplace.
- `--simplify` : Whether to simplify onnx. Not support in end to end export.
- `--end2end` : Whether to export end to end onnx model. Only support onnxruntime and TensorRT >= 8.0.0 .
- `--trt-version` :  Export onnx for TensorRT version. Support : 7 or 8.
- `--ort` : Whether to export onnx for onnxruntime backend.
- `--with-preprocess` : Whether to export preprocess with bgr2rgb and normalize (divide by 255)
- `--topk-all` : Topk objects for every image.
- `--iou-thres` : IoU threshold for NMS algorithm.
- `--conf-thres` : Confidence threshold for NMS algorithm.
- `--device` : Export device. Cuda device : 0 or 0,1,2,3 ... , CPU : cpu .

## Pico deployment and export

The exporter calls `PicoBlock.switch_to_deploy()` after loading the checkpoint
in evaluation mode. It fuses the four small parallel depthwise branches into one
3x3 convolution and folds the final pointwise convolution's BN. The sequential
1x7/7x1 depthwise pair is composed into one 7x7 depthwise convolution in the
deployment model and ONNX graph, using the per-channel outer product of the kernels.
The multi-scale channel split,
concatenation, first BN, activation and outer residual remain. The entire block
cannot collapse into a single convolution because its residual follows an
activation. Conversion preserves device/dtype and supports repeated calls.

To convert a backbone in Python:

```python
from yolov6.utils.checkpoint import load_checkpoint

model = load_checkpoint('pico.pt', map_location='cpu', fuse=False)
model.eval()
model.backbone.switch_to_deploy()
```

The checkpoint is loaded before conversion; its file is not rewritten. The
existing `model.backbone.fuse()` method is an alias for this conversion.

New Pico models default to `model.neck.upsample_type='nearest'`, producing ONNX
Resize nodes without GridSample. Full-model export supports PyTorch 1.10 and
defaults to opset 13:

```bash
python deploy/ONNX/export_onnx.py --weights pico.pt \
    --img-size 416 --device cpu --dynamic-batch --simplify
```

Setting `upsample_type='dysample'` enables learned sampling. Such models include
`grid_sample`, requiring PyTorch 1.13 or newer, `onnx>=1.12`, and ONNX opset 16
or newer. Use a separate export environment if training runs on PyTorch 1.10:

```bash
python deploy/ONNX/export_onnx.py --weights pico.pt \
    --img-size 416 --device cpu --opset 16 --dynamic-batch --simplify
```

This writes `pico.onnx`, containing decoded predictions `[B,N,5+nc]`. The default
export does not include NMS. When DySample is enabled, the runtime must support
ONNX `GridSample`; check the target TensorRT/backend version separately. `Detect.export=True` is not
needed: the existing model export path returns the decoded tensor.

The exporter inspects the loaded checkpoint's modules, so existing DySample
checkpoints still select opset 16 even after changing the default config.

To save a new 7x7-fused ONNX file from the existing DySample checkpoint with a
64-channel neck/head:

```bash
python deploy/ONNX/export_onnx.py \
    --weights weights/yolov6_pico_64ch/pico_checkpoint.pt \
    --output weights/yolov6_pico_64ch/pico_7x7.onnx \
    --img-size 416 --device cpu --opset 16 --dynamic-batch --simplify
```

Previously exported ONNX files retain their saved graph; re-export to apply
this conversion. Older factorized deployment checkpoints are also converted.

## Download

* [YOLOv6-N](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.onnx)
* [YOLOv6-T](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.onnx)
* [YOLOv6-S](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx)
* [YOLOv6-M](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.onnx)
* [YOLOv6-L-ReLU](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l_relu.onnx)
* [YOLOv6-L](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.onnx)


## End2End export

Now YOLOv6 supports end to end detect for onnxruntime and TensorRT !

If you want to deploy in TensorRT, make sure you have installed TensorRT !

### onnxruntime backend
#### Usage

```bash
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --end2end \
    --ort
```

You will get an onnx with **NonMaxSuppression** operator .

### TensorRT backend (TensorRT version == 7.2.3.4)
#### Usage
```bash
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --end2end \
    --trt-version 7
```
You will get an onnx with **[BatchedNMSDynamic_TRT](https://github.com/triple-Mu/TensorRT/tree/main/plugin/batchedNMSPlugin)** plugin .


### TensorRT backend (TensorRT version>= 8.0.0)

#### Usage

```bash
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --end2end \
    --trt-version 8
```

You will get an onnx with **[EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)** plugin .

### Outputs Description

The onnx outputs are as shown :

<img src="https://user-images.githubusercontent.com/92794867/176650971-a4fa3d65-10d4-4b65-b8ef-00a2ff13406c.png" height="300px" />

```num_dets``` means the number of object in every image in its batch .

```det_boxes``` means topk(100) object's location about [`x0`,`y0`,`x1`,`y1`] .

```det_scores``` means the confidence score of every topk(100) objects .

```det_classes``` means the category of every topk(100) objects .


You can export TensorRT engine use [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-ovr) tools.
#### Usage
For both TensorRT-7 and TensorRT-8  `trtexec`  tool is avaiable.
``` shell
trtexec --onnx=yolov6s.onnx \
        --saveEngine=yolov6s.engine \
        --workspace=8192 # 8GB
        --fp16 # if export TensorRT fp16 model
```

## Evaluate TensorRT model's performance

When we get the TensorRT model, we can evalute its performance by:
```
python deploy/ONNX/eval_trt.py --weights yolov6s.engine --batch-size=1 --data data/coco.yaml
```

## Dynamic Batch Inference

YOLOv6 support dynamic batch export and inference, you can refer to:

[export ONNX model with dynamic batch ](YOLOv6-Dynamic-Batch-onnxruntime.ipynb)

[export TensorRT model with dynamic batch](YOLOv6-Dynamic-Batch-tensorrt.ipynb)
