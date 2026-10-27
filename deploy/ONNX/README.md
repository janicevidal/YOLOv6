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
- `--raw-head` : Export Pico P3/P4/P5 maps without anchor generation or box decoding; mutually exclusive with `--end2end`.
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

Setting `model['neck']['upsample_type'] = 'sni'` selects soft nearest-neighbor
interpolation: each 2x top-down upsampling is multiplied by 1/4. SNI adds no
learned parameters and exports as standard ONNX Resize and Mul at opset 13,
including with `--raw-head`. Train with the SNI config before exporting its
checkpoint; export uses the modules stored in the checkpoint.

Setting `upsample_type='dysample'` enables learned sampling. Such models include
`grid_sample`, requiring PyTorch 1.13 or newer, `onnx>=1.12`, and ONNX opset 16
or newer. Use a separate export environment if training runs on PyTorch 1.10:

```bash
python deploy/ONNX/export_onnx.py --weights pico.pt \
    --img-size 416 --device cpu --opset 16 --dynamic-batch --simplify
```

This writes `pico.onnx`, containing decoded predictions `[B,N,5+nc]`. The default
export does not include NMS. When DySample is enabled, the runtime must support
ONNX `GridSample`; check the target TensorRT/backend version separately.

To remove anchor generation and box decoding from the Pico ONNX graph, export
the three head maps instead:

```bash
python deploy/ONNX/export_onnx.py --weights pico.pt \
    --output pico_raw.onnx --img-size 416 --device cpu \
    --raw-head --dynamic-batch --simplify
```

For a single class at 416x416, the outputs are `p3=[B,5,52,52]`,
`p4=[B,5,26,26]`, and `p5=[B,5,13,13]`, at strides 8, 16 and 32.
Channels are `[class probabilities..., l, t, r, b]`; probabilities already
include sigmoid. The four distances are raw LTRB offsets in feature-grid units,
with no DFL, clipping or extra activation. There is no objectness channel.
The deployment postprocessor must decode the boxes and perform NMS. For grid
position `(x,y)` at stride `s`, the pixel coordinates are:

```text
x1 = (x + 0.5 - l) * s       y1 = (y + 0.5 - t) * s
x2 = (x + 0.5 + r) * s       y2 = (y + 0.5 + b) * s
```

`Model.export` only suppresses the outer feature-map return; the separate
`model.detect.export` flag selects the head's early return before
`generate_anchors()` and `dist2bbox()`. `--raw-head` sets both flags, assigns
three output names and applies dynamic-batch axes to all outputs. Default
export explicitly keeps Pico decoding enabled. `--raw-head` supports PicoHead
only and cannot be combined with `--end2end`, which requires decoded boxes.
This flag does not remove DySample sampling grids when that upsampler is used.

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

## ONNX Runtime folder inference and COCO evaluation

`infer_onnx.py` runs without PyTorch. Install the CPU dependencies with:

```shell
pip install numpy opencv-python onnxruntime pycocotools
```

For CUDA, install a CUDA/cuDNN-compatible `onnxruntime-gpu` instead of
`onnxruntime` and pass `--device 0`. CPU is the default. The script supports:

- Pico `--raw-head`: three `[B,nc+4,H,W]` maps, channels `[scores,l,t,r,b]`.
  It constructs grid centers at `0.5`, decodes LTRB with each level's stride,
  and applies class-aware NMS. Class scores are already probabilities.
- Decoded YOLOv6: `[B,N,5+nc]`, containing pixel `xywh`, objectness, class scores.
- `--end2end --ort`: `num_dets`, `det_boxes`, `det_scores`, `det_classes`.
  Supply `--num-classes` or `--category-ids`/`--names` to specify class count.
  NMS is already in the graph; its export-time thresholds and topk still apply.
  TensorRT-specific NMS plugins and `--with-preprocess` models are unsupported.

Example for the single-class Pico raw-head model:

```shell
python deploy/ONNX/infer_onnx.py \
    --model /path/to/last_ckpt.onnx \
    --source /path/to/images \
    --names person --save-txt \
    --output-dir runs/onnx/person_demo
```

The script recursively processes images and saves the following, preserving
source subdirectories:

- `images/`: original-size annotated images. `--vis-conf` defaults to 0.25;
  `--no-save-images` skips image writing.
- `labels/*.jpg.txt` (with `--save-txt`): one line per detection, in the format
  `class_id score x1 y1 x2 y2`, using zero-based class indices and original-image
  pixel coordinates. Keeping the image extension avoids JPG/PNG stem collisions.
- `detections.json`: per-image detections, including empty results, names,
  dimensions and original-image `bbox_xyxy`.
- `predictions.json`: COCO result entries with `image_id`, `category_id`,
  top-left pixel `bbox=[x,y,width,height]`, and `score`.
- `metrics.json`: model/runtime/settings/counts/timing, plus COCO metrics when
  annotations are supplied.

Add the original COCO validation annotation JSON to evaluate:

```shell
python deploy/ONNX/infer_onnx.py \
    --model /path/to/last_ckpt.onnx \
    --source /data/zhangxiaoshuai/HumanDetection/coco_person/images/val2017 \
    --annotations /data/zhangxiaoshuai/HumanDetection/coco_person/annotations/instances_val2017_person.json \
    --category-ids 1 --conf-thres 0.001 --iou-thres 0.65 \
    --no-save-images --output-dir runs/onnx/person_val
```

Evaluation uses official `pycocotools.COCOeval(..., 'bbox')`, including its
IoUs 0.50:0.05:0.95, area ranges, crowd handling and maxDets `[1,10,100]`.
All annotation images participate, including images with zero predictions.
Missing images cause an error rather than silently evaluating an incomplete
validation set. Image IDs come from annotation `file_name` matching, so numeric
filenames are not required. Without annotations, `image_id` is the relative
filename; supply annotations to produce IDs usable with that validation set.

`--category-ids` maps model class indices to annotation category IDs in order.
For a person-only model on COCO, model class 0 maps to category ID 1. If the
annotation category count equals the model class count, the default mapping is
sorted category IDs; explicitly specify the list for a different training class
order. Only the mapped categories are evaluated, so person-only evaluation on
full COCO annotations reports person AP. It is not the 80-category COCO mAP.

The inference threshold defaults to 0.25 without annotations and 0.001 with
annotations. Drawing uses the independent `--vis-conf`, preserving low-scoring
predictions for AP. An end-to-end graph exported at a high score threshold cannot
recover discarded boxes by lowering this script's threshold.

Input size and static batch are read from ONNX. A `[1,3,416,416]` model processes
a folder one image at a time; this is still folder batch processing. To use
`--batch-size 8`, export with `--dynamic-batch`. A fixed batch greater than one
uses padding for the final incomplete batch and discards padded results. Dynamic
spatial dimensions require `--img-size HEIGHT WIDTH`.

Preprocessing supports `--resize-mode letterbox` (default), preserving aspect
ratio with padding 114, and `--resize-mode stretch`, directly resizing to the
model input without padding. Both use bilinear interpolation, BGR-to-RGB, NCHW
and division by 255. Stretch restores x/y coordinates using separate scale
factors; letterbox removes padding before restoring coordinates. The chosen
mode is recorded in `metrics.json` and applies to inference and COCO evaluation.
For example:

```shell
python deploy/ONNX/infer_onnx.py \
    --model /path/to/last_ckpt.onnx --source /path/to/images \
    --resize-mode stretch --names person --output-dir runs/onnx/person_stretch
```

These fixed-size paths can differ from the PyTorch validator's rectangular
batching/two-stage resizing; matching official COCO metrics does not guarantee
identical AP across different preprocessing settings.

Validation used ONNX Runtime 1.11.1 on an Intel Xeon Silver 4216 CPU with two
intra-op threads, batch 1 and 416x416 input. The supplied
`yolov6_pico_small_wiou_bifpan_stal_fuseab_sni_refine/weights/last_ckpt.onnx`
was evaluated on all 2,693 images in the person validation JSON shown above,
with confidence 0.001, NMS IoU 0.65 and max-det 300. Official person AP50 was
0.5509901003 and AP50:95 was 0.2922677614. Raw decoding/NMS matched the project's
PyTorch postprocessor on actual ONNX output. Additional checks covered dynamic
batch for all three formats, fixed-batch tail padding, nested/duplicate-stem
filenames, category/image ID mapping and perfect/partial/empty COCO predictions.
Sample-image inference saved annotated images and TXT/JSON results successfully.

## Compare ONNX Runtime with training validation

`compare_onnx.py` runs matching `.onnx` and `.pt` weights through a shared
YOLOv6 validation pipeline. Unlike `infer_onnx.py`, it requires PyTorch and the
project dependencies as well as ONNX Runtime and pycocotools. Run from the
repository root:

```shell
weights_dir=/home/zhangxiaoshuai/Checkpoint/HumanDetetct/checkpoints/yolov6_pico_small_wiou_bifpan_stal_fuseab_sni_refine_resume_lr_fp32/weights

# Compare the supplied fixed 1x3x416x416 ONNX against PyTorch on the same tensors.
python deploy/ONNX/compare_onnx.py \
    --model "$weights_dir/best_ckpt.onnx" \
    --data data/coco_person_h50.yaml \
    --img-size 416 --batch-size 256 --infer-batch-size 1 \
    --conf-thres 0.03 --iou-thres 0.65 \
    --output-dir runs/onnx/best_ckpt_compare/square

# Match training validation's rectangular image shapes, using a separate export.
python deploy/ONNX/compare_onnx.py \
    --model "$weights_dir/best_ckpt.onnx" \
    --data data/coco_person_h50.yaml \
    --img-size 416 --batch-size 256 --infer-batch-size 1 \
    --rect --export-dynamic \
    --conf-thres 0.03 --iou-thres 0.65 \
    --output-dir runs/onnx/best_ckpt_compare/rect
```

`--weights` defaults to the `.pt` beside the ONNX. The native model is the
checkpoint's EMA when present (stripped checkpoints already store EMA as
`model`). It stays FP32, eval mode, without deployment fusion, as in training
validation. Raw Pico outputs are decoded with the project's own
`generate_anchors` and `dist2bbox`; decoded `[B,N,5+nc]` ONNX is also accepted.
Models with embedded preprocessing or NMS cannot use this comparison.

Both backends share these operations:

- `TrainValDataset`: native integer resize dimensions, `INTER_AREA` when
  downscaling, validation letterbox (`scaleup=False`, fill 114), BGR-to-RGB,
  CHW, FP32 and `/255`. Each input tensor is reused by both backends.
- Project `non_max_suppression`, with training defaults confidence 0.03,
  IoU 0.65, `multi_label=True`, max-det 300, and the same execution device.
- `Evaler.convert_to_coco_format`: separate x/y resize gains, integer padding,
  clipping, XYWH conversion, bbox rounding to 3 decimals and scores to 5.
- Official COCO bbox evaluation with the same complete image/category lists,
  IoUs, crowd handling, area ranges and maxDets `[1,10,100]`.

Square mode shares the native resize/letterbox implementation, but fixes the
target shape to fit the supplied ONNX. Training uses `rect=True, pad=0.5` in
`Trainer.get_data_loader`, even when training augmentation itself uses
`rect=False`. For 416 input, the rectangular validation shapes can include a
448 dimension. A static 416 ONNX cannot reproduce those inputs.
`--rect --export-dynamic` therefore writes `dynamic_raw.onnx` inside its output
directory using the existing Pico deployment conversions and dynamic B/H/W.
It evaluates that new model and preserves the original ONNX. `--rect` alone
also accepts a previously exported dynamic model.

`--batch-size` controls the aspect-ratio groups used to calculate rectangular
shapes. Match the training validation loader: `train_batch_size/world_size*2`
(256 for this experiment). `--infer-batch-size` controls the execution batches
inside each group. The default 1 accommodates the supplied static ONNX and
keeps the two backends' execution batch sizes equal. Different execution batch
sizes can introduce small floating-point differences; they do not change the
recorded rectangular grouping. A static batch greater than one pads incomplete
batches identically for both backends, then discards the padded predictions.
Use `--config-file` to carry over `data_aug`
if the training config includes preprocessing settings such as `shrink_size`.
Training configs with custom `eval_params` thresholds must also be matched with
the corresponding CLI thresholds.

`--data` supplies `val`, `anno_path`, `nc`, `names`; `--source` and
`--annotations` can override paths. The example uses the existing
`coco_person_h50.yaml` because the run's original `coco_person.yaml` points to
an obsolete dataset location. Every annotation image must be present in the
loader. `--category-ids 1` explicitly selects person class 0 -> category 1;
the default is sorted category IDs when category count equals model class
count. `--limit 8` runs a marked subset for debugging.

Each run saves `pytorch_predictions.json`, `onnxruntime_predictions.json` and
`comparison.json`. The report contains all 12 COCO metrics, ONNX-minus-PyTorch
metric deltas, detection counts, decoded prediction mean/max errors, per-image
forward/decode and NMS/restore times, input shape counts, model hashes,
dataset paths, settings and runtime versions. Export time, warmup, COCOeval and
result writing are excluded from inference times. ORT includes host copies and
external raw-head decoding. The timings describe this serial comparison, not
an isolated throughput benchmark. CPU is the default; `--device 0` runs both
backends and NMS on the selected GPU, with a compatible `onnxruntime-gpu`.

Validation on the example `best_ckpt` used all 2,693 person validation images,
FP32, execution batch 1, grouping batch 256, CPU two threads (Intel Xeon Silver
4216), PyTorch 1.10.1 and ONNX Runtime 1.11.1. Confidence was 0.03 and NMS IoU
0.65. The supplied original static ONNX and a separately exported dynamic ONNX
gave the following results:

| Input mode | Backend | AP50 | AP50:95 | Forward + decode ms/image | NMS + restore ms/image |
| --- | --- | --- | --- | --- | --- |
| Fixed 416x416 | Native PyTorch | 0.64438743 | 0.37987842 | 44.058 | 4.104 |
| Fixed 416x416 | Supplied ONNX | 0.64438734 | 0.37988135 | 17.815 | 4.431 |
| Training rectangles | Native PyTorch | 0.64834073 | 0.38212971 | 44.058 | 3.983 |
| Training rectangles | New dynamic ONNX | 0.64834033 | 0.38211488 | 16.912 | 3.874 |

The checkpoint's saved training validation metrics were AP50 0.6483407276 and
AP50:95 0.3821150420. The dynamic ONNX AP50:95 differs by approximately
-0.0000001584. CPU microbatch validation and the original GPU batch validation
can differ slightly at numerical/NMS/IoU boundaries. Within the shared input
comparisons, ONNX-minus-PyTorch AP50:95 was +0.0000029316 (square) and
-0.0000148305 (rectangular). The original fixed input changes AP more than the
observed backend/export numerical differences.

The full training rectangles were 448x320, 448x352, 448x448, 352x448, 320x448
and 288x448. Decoded box maximum absolute error was below 0.0035 input pixels
and class-score maximum error below 0.000022 for both full comparisons.
Additional checks covered dynamic batch 2 with a final incomplete batch,
fixed batch 2 tail padding, and agreement between the supplied static graph
and the generated dynamic graph at 416x416. Pre-commit checks passed.

## Yolo-Fastest person ONNX inference and COCO evaluation

`infer_yolo_fastest_onnx.py` uses the person anchors and postprocessing from
`detect_online`. It shares folder traversal, output writers and official
COCOeval with `infer_onnx.py`. Dependencies are NumPy, OpenCV, ONNX Runtime
and, for COCO evaluation, pycocotools. PyTorch is unnecessary.

The supplied `yolo_fastest_person0719.onnx` has FP32 input `[1,3,352,352]`
and six activated NCHW outputs:

| Stride | Regression | Objectness | Person class |
| --- | --- | --- | --- |
| 16 | `out_reg_2 [1,12,22,22]` | `out_obj_2 [1,3,22,22]` | `out_cls_2 [1,1,22,22]` |
| 32 | `out_reg_3 [1,12,11,11]` | `out_obj_3 [1,3,11,11]` | `out_cls_3 [1,1,11,11]` |

The graph already applies sigmoid to reg/obj and softmax to cls. Decoding
uses `xy=(2*reg_xy-0.5+grid)*stride`, `wh=(2*reg_wh)^2*anchor`, score `obj*cls`.
Anchors are fixed to the deployment table in
`detect_online/source/YXAIHumandtMisc.cpp`:

| Stride | Anchors (width, height), in input pixels |
| --- | --- |
| 16 | `(9.70,22.40), (28.50,63.70), (52.10,146.70)` |
| 32 | `(103.41,238.33), (173.57,91.55), (246.92,273.87)` |

Boxes are restored to the original image, with letterbox padding subtracted
when applicable, then corners truncate toward zero to int32 before NMS.
The default preserves out-of-image coordinates. `--clip-boxes` optionally
clips before NMS. The source's single-person, square-input assumptions apply.
Defaults from `YXAIHumandtInner.h` / `YXAIHumandt.h` are confidence **0.30**,
NMS IoU **0.25** and at most **10** detections per image.

Preprocessing supports all four combinations:

- `--resize-mode stretch` (default): direct bilinear resize; independent x/y
  box restoration. CHW float32 multiplication by `1/255` follows the deployment.
- `--resize-mode letterbox`: aspect-preserving resize and padding 114; reuses
  `infer_onnx.py` preprocessing, including division by 255.
- `--color-order bgr` (default) or `--color-order rgb`: choose model input
  channels. BGR/direct resize follows the deployment caller's behavior.

The private bilinear resize and channel-conversion implementations are absent
from `detect_online`. OpenCV `INTER_LINEAR` is used, and BGR retention follows
the CV_BGR caller's use of `uint8RGBToCHWFloatRGB`. Pixel-level equivalence to
those private routines is unverified; C++ postprocessing was checked separately.

Folder inference, saving annotated images, JSON and optional TXT:

```shell
python deploy/ONNX/infer_yolo_fastest_onnx.py \
    --model yolo_fastest_person0719.onnx \
    --source data/images --names person --save-txt \
    --resize-mode stretch --color-order bgr \
    --threads 2 --output-dir runs/onnx/fastest_person_demo
```

For RGB letterbox, replace the preprocessing options with
`--resize-mode letterbox --color-order rgb`.

Official COCO person evaluation with lower confidence and a higher detection
limit to preserve recall:

```shell
python deploy/ONNX/infer_yolo_fastest_onnx.py \
    --model yolo_fastest_person0719.onnx \
    --source /data/zhangxiaoshuai/HumanDetection/coco_person/images/val2017 \
    --annotations /data/zhangxiaoshuai/HumanDetection/coco_person/annotations/instances_val2017_person.json \
    --category-ids 1 --img-size 352 --conf-thres 0.001 --max-det 300 \
    --threads 2 --no-save-images \
    --output-dir runs/onnx/fastest_person_val
```

Even with `--annotations`, defaults remain **0.30 / 0.25 / 10**; remove
`--conf-thres 0.001 --max-det 300` to evaluate that deployment operating point.
`metrics.json` records anchors/source, preprocessing, thresholds, max-det,
model hash, runtime, timing and all 12 COCO metrics. Predictions are saved to
`predictions.json`, with per-image results in `detections.json`.

Full BGR/direct-resize validation used all **2,693** images / **11,004** person
annotations, FP32 **352x352**, batch 1, ONNX Runtime **1.11.1**, Intel Xeon
Silver 4216 CPU with **2 threads**. The ONNX SHA256 is
`21e6d07b0bf2b69248ab7e08048ccfffa4f2b48813994751000e9a5da99956e3`.

| Confidence | NMS IoU | max-det | AP50 | AP50:95 | AR@100 |
| --- | --- | --- | --- | --- | --- |
| 0.30 | 0.25 | 10 | 0.365887 | 0.202666 | 0.235836 |
| 0.001 | 0.25 | 300 | 0.453324 | 0.228509 | 0.298302 |

These are single-class person metrics. COCOeval's maxDets=100 cannot recover
boxes removed by the application's top10 limit. Full reports and predictions
are under `runs/onnx/fastest_detect_online352/deployment_defaults/` and
`runs/onnx/fastest_detect_online352/coco_low_conf/`. The original C++ decoder
and NMS matched Python exactly on 28 real images with three settings each
(**84** comparisons). Cleanup also checked all four preprocessing combinations,
letterbox coordinate restoration, batch 2, empty predictions and sample-image
inference. Historical evaluation artifacts remain in `runs/`.

## Dynamic Batch Inference

YOLOv6 support dynamic batch export and inference, you can refer to:

[export ONNX model with dynamic batch ](YOLOv6-Dynamic-Batch-onnxruntime.ipynb)

[export TensorRT model with dynamic batch](YOLOv6-Dynamic-Batch-tensorrt.ipynb)
