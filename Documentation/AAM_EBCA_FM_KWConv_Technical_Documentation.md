# AAM–EBCA–FM–KWConv Green-Apple Detection System

## Formula, architecture, code-connection, training, depth, and tracking documentation

**Document basis:** `Final PPT.pptx`, especially slides 2–11, checked against the completed source code, trained checkpoint, manifests, logs, and result files.
**Canonical line-reference file:** `Codeing/aam_ebca_fm_kwconv_main.py`
**Primary research notebook:** `Codeing/AAM, EBCA, FM, KWConv 100.ipynb`

> Scientific interpretation: the detection validation labels are automatically generated pseudo-labels. The AdaBins-derived values are relative/estimated depth and have not been camera-calibrated as physical metres.

## 1. As-built end-to-end process

The implemented process is:

1. Read the 1920×1080, 59.94-FPS orchard video containing 3,825 source frames.
2. Build a timestamp-accurate 80-FPS timeline of 5,105 images. Because the camera did not record 80 unique frames per second, repeated source-frame indices are recorded rather than inventing new camera content.
3. Select 800 uniformly distributed timeline frames.
4. Produce ten controlled variants of every selected frame: horizontal flip, vertical flip, brighter, darker, 90° rotation, 180° rotation, 270° rotation, small random rotation, motion blur, and Gaussian noise.
5. Keep every original and its ten derivatives in the same data partition. The grouped 70/10/20 split is 560/80/160 source groups, or 6,160/880/1,760 images.
6. Generate single-class `green_apple` pseudo-labels using COCO `apple` proposals from full-frame and tiled YOLO inference, followed by non-maximum suppression.
7. Extract YOLOv8 P3/P4/P5 features and apply AAM → EBCA → FM → KWConv before the three-scale Detect head.
8. Detect apples and confidence scores.
9. Estimate an AdaBins depth map, calculate robust per-box depth, and apply the apparent-scale correction used by the published result tables.
10. Associate detections with the depth-aware ByteTrack-style tracker to generate persistent IDs.
11. Count a confirmed track once when its centre crosses the configured line. The filtered presentation result contains 96 duplicate-safe crossings.

### Slide-2 architecture clarification

Slide 2 is a conceptual overview, not an exact software graph. In the trained implementation:

- AAM, EBCA, FM, and KWConv form one enhancement neck immediately before the existing Ultralytics Detect head.
- AAM acts on P4; EBCA exchanges P3/P4 information; FM refines P4; KWConv independently refines P3, P4, and P5.
- AdaBins is post-detection processing on the RGB frame. It is not inserted between the YOLO neck and Detect head.
- ByteTrack consumes detector boxes, confidence, and corrected relative depth after detection.
- The source is 1920×1080, while the completed low-VRAM detector run used `imgsz=640` for training/inference tensor resizing.

## 2. Data characteristics and features extracted

| Stage | Extracted information | Purpose | Code |
|---|---|---|---|
| Timeline | Timestamp, source-frame index, repeated-frame flag | Exact temporal accounting | `extract_80fps_timeline`, lines 360–428 |
| Uniform sampling | Global temporal coverage of the video | Avoid concentration in one short interval | `pick_uniform_800`, lines 431–459 |
| Augmentation | Orientation, illumination, rotation, blur, and sensor-noise variation | Improve visual robustness | `augment_picked_800`, lines 496–545 |
| Grouped split | Source-frame family membership | Prevent original/augmentation leakage | `build_grouped_dataset_split`, lines 566–613 |
| Pseudo-labels | Apple box centre, width, height, and proposal confidence | Train the one-class detector | `auto_label_after_split`, lines 728–801 |
| AAM | Colour/channel response, local shape edges, and scale context | Attribute-aware P4 representation | `AAM`, lines 858–874 |
| EBCA | High-resolution P3 detail and deep P4 semantics | Multi-scale overlap handling | `EBCA`, lines 877–896 |
| FM | Local-to-global contextual evidence | Robustness to partial occlusion | `FM`, lines 899–919 |
| KWConv | 3×3, 5×5, and 7×7 receptive-field responses | Adaptive fruit-size handling | `KWConv`, lines 922–935 |
| AdaBins | Dense monocular depth and robust box statistic | Near/far association cue | lines 1169–1272 |
| Tracker | IoU, centre motion, confidence tier, and relative-depth consistency | Persistent ID and duplicate-safe count | lines 1276–1462 |

The augmentation seed is 42. The completed detector training seed stored in the checkpoint is 0; these are separate random processes.

## 3. Attribute Attention Module (AAM) — PPT slide 3

### PPT formula

\[
F_{AAM}=F\odot\sigma(\alpha F_{colour}+\beta F_{shape}+\gamma F_{scale})
\]

Here, `F` is the P4 input feature map, `σ` is sigmoid, `⊙` is element-wise multiplication, and `α, β, γ` are learned fusion weights.

### Implemented branches

- **Colour branch:** global average pooling → 1×1 channel reduction → SiLU → 1×1 channel restoration → spatial expansion.
- **Shape branch:** channel-wise/depthwise 3×3 convolution.
- **Scale branch:** 3×3 average pooling followed by 1×1 convolution.
- **Fusion:** `softmax([w_colour,w_shape,w_scale])` ensures normalized learned weights.

The exact implemented relation is:

\[
\boldsymbol{\alpha}=softmax(\mathbf{w}),\qquad
F'=F\odot\sigma(\alpha_1 C(F)+\alpha_2 S(F)+\alpha_3 R(F)).
\]

**Code:** `AAM`, lines 858–874. This module is consistent with the slide-3 formula.

## 4. Efficient Bidirectional Cross-Attention (EBCA) — PPT slides 4–6

### Conceptual PPT equations

Slides 4–6 describe dense scaled dot-product attention:

\[
Q=P_3W_Q,\quad K=P_4W_K,\quad V=P_4W_V,
\]

\[
S=QK^T,\qquad A=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right),\qquad F=AV.
\]

Bidirectional and multi-head forms are presented as:

\[
F_{P3}=Attention(Q_{P3},K_{P4},V_{P4}),
\]

\[
F_{P4}=Attention(Q_{P4},K_{P3},V_{P3}),
\]

\[
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V),
\]

\[
MultiHead=Concat(head_1,\ldots,head_h)W^O.
\]

The PPT additionally illustrates eight heads, positional encoding, a CSP block, and FPN integration.

### Formula actually used by the trained checkpoint

The trained model uses a memory-efficient element-wise bidirectional gate. It avoids the quadratic spatial `QKᵀ` matrix, which is important on the 4-GB RTX 3050 Ti.

For P4 → P3:

\[
U=Resize(W_{43}*P_4),
\]

\[
G_3=\sigma\left(\frac{(W_{q3}*P_3)\odot U}{\sqrt{C_3}}\right),
\]

\[
P'_3=BN(P_3+G_3\odot U).
\]

For P3 → P4:

\[
D=AdaptivePool(W_{34}*P_3),
\]

\[
G_4=\sigma\left(\frac{(W_{q4}*P_4)\odot D}{\sqrt{C_4}}\right),
\]

\[
P'_4=BN(P_4+G_4\odot D).
\]

**Code:** `EBCA`, lines 877–896.

### Traceability decision

The dense eight-head PPT equations are conceptual background; they are not the computation stored in `best.pt`. Replacing the trained EBCA with dense multi-head attention would change tensor parameters, invalidate the checkpoint, and require complete retraining. The documentation therefore records the actual efficient gate instead of claiming that the checkpoint contains unimplemented operations.

## 5. Focal Modulation (FM) — PPT slides 7–9

### PPT equations

The slides define query/context projections, multi-level depthwise context, gating, modulation, and output transformation:

\[
q(x)=W_qX,\qquad X^1=W_cX,
\]

\[
f_l(x)=DWConv_l(X^l),
\]

\[
g_l(x)=\sigma(W_{gl}f_l(x)),
\]

\[
m(x)=\sum_{l=1}^{L}g_l(x)f_l(x),
\]

\[
y=q(x)\odot m(x),\qquad Z=W_oy.
\]

### Implemented computation

The code uses three hierarchical depthwise kernels of 3×3, 5×5, and 7×7, plus one global context branch. Gates are normalized jointly with softmax:

\[
Q=Conv_{1\times1}(X),\qquad G=softmax(Conv_{1\times1}(X)),
\]

\[
C_l=DWConv_l(C_{l-1}),\quad C_0=X,
\]

\[
M=\sum_{l=1}^{L}G_l\odot C_l+G_{L+1}\odot GAP(X),
\]

\[
Z=Conv^{out}_{1\times1}\left(Q\odot Conv^{mod}_{1\times1}(M)\right).
\]

**Code:** `FM`, lines 899–919. The implementation preserves the focal-modulation idea while making scale gates compete through a normalized softmax.

## 6. Kernel Warehouse Convolution (KWConv) — PPT slides 10–11

### PPT equations

For channel `c`, global average pooling is:

\[
Z_c=\frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}X_c(i,j).
\]

Kernel selection and assembly are:

\[
\boldsymbol{\alpha}=softmax(WZ),\qquad E=\{e_1,e_2,\ldots,e_n\},
\]

\[
K=\sum_{i=1}^{n}\alpha_ie_i,\qquad Y=K*X.
\]

### Implemented computation

The warehouse consists of learned 3×3, 5×5, and 7×7 convolution branches. A global pooled selector produces one normalized weight per branch and sample:

\[
\boldsymbol{\alpha}=softmax(Conv_{1\times1}(GAP(X))),
\]

\[
Y=SiLU\left(BN\left(\sum_{i=1}^{3}\alpha_i(e_i*X)\right)\right).
\]

Because convolution is linear and each `αᵢ` is spatially constant for one sample, mixing branch responses is equivalent to applying the assembled kernel mixture, with the implementation avoiding explicit kernel materialization.

**Code:** `KWConv`, lines 922–935.

## 7. Code connection among AAM, EBCA, FM, KWConv, and Detect

The exact connection is:

```text
YOLO P3 ───────────────────┐
                          ├─ EBCA ─ P3' ─ KWConv3 ─┐
YOLO P4 ─ AAM ────────────┘                       │
                    EBCA P4' ─ FM ─ KWConv4 ──────┼─ Enhanced Detect(P3'',P4'',P5')
YOLO P5 ─────────────────────── KWConv5 ──────────┘
```

Equivalent pseudocode:

```python
p4 = AAM(p4)
p3, p4 = EBCA(p3, p4)
p4 = FM(p4)
p3, p4, p5 = KWConv(p3), KWConv(p4), KWConv(p5)
predictions = Detect([p3, p4, p5])
```

Code connections:

- `ProposedEnhancementNeck`, lines 938–952.
- `EnhancedDetectHead`, lines 955–999.
- `patch_yolov8_detection_head`, lines 1039–1058.
- Custom trainer integration, lines 1071–1125.

The completed checkpoint contains an `EnhancedDetectHead`. It has 40,151,203 parameters in total; the enhancement neck contributes 29,015,216 parameters. The feature-channel inputs are P3=128, P4=256, and P5=512.

## 8. AdaBins depth processing

The RGB frame is resized to 928×512 before official KITTI-pretrained AdaBins inference, and its output is resized back to source resolution. Within each detection, a 15% margin excludes unstable box edges. Positive finite values are sorted, 10% is trimmed from each tail, and the median becomes the raw per-instance depth:

\[
d_{raw}=median\left(trim_{10\%}(D[B_{inner}])\right).
\]

For the published corrected results, equivalent box diameter is:

\[
s_i=\sqrt{w_ih_i}.
\]

The frame reference diameter is the median `sᵢ` among detections whose confidence is at least 0.50. The correction is:

\[
f_i=clip\left(\left(\frac{s_{ref}}{s_i}\right)^{0.65},0.5,2.5\right),
\]

\[
d_i=d_{raw,i}f_i.
\]

Code locations:

- `AdaBinsDepth`, lines 1169–1205.
- `robust_instance_depth`, lines 1208–1225.
- `apparent_scale_corrected_depths`, lines 1235–1272.

These values are suitable as a relative association cue. They must not be reported as verified metres without camera/depth calibration.

## 9. Depth-aware ByteTrack-style association and counting

The tracker first predicts each box using its smoothed centre velocity. Detections are divided into high-confidence (`score ≥ 0.50`) and low-confidence (`0.10 ≤ score < 0.50`) sets, following the two-stage ByteTrack idea.

For a valid IoU/depth-gated pair, the implemented cost is:

\[
C=0.65(1-IoU)+0.20C_{depth}+0.15C_{motion}.
\]

The normalized depth difference is:

\[
\delta_d=\frac{|d_t-d_j|}{\max(|d_t|,|d_j|,10^{-6})}.
\]

Associations with `δd > 0.40` are rejected; otherwise `C_depth=δd/0.40`. Hungarian assignment chooses the minimum-cost pairs. Confirmed tracks require at least three hits and may remain unmatched for up to 30 frames.

For duplicate-safe counting, the side of line endpoints `a,b` for centre `p` is the 2-D cross product:

\[
side(p)=(b_x-a_x)(p_y-a_y)-(b_y-a_y)(p_x-a_x).
\]

A confirmed track is counted once when consecutive side values change sign.

Code locations:

- Detection/track state, lines 1229–1306.
- `DepthAwareByteTracker`, lines 1309–1371.
- `UniqueLineCounter`, lines 1374–1397.
- End-to-end inference, lines 1400–1462.

## 10. Completed training configuration

The configuration must be reported from the checkpoint/logs, not copied from another paper:

| Setting | Completed project value |
|---|---:|
| Base detector | COCO-pretrained YOLOv8s |
| Target class | `green_apple` |
| Epochs | 300 |
| Input tensor size | 640×640 |
| Batch size | 1 |
| Workers | 4 |
| Optimizer | `auto`, resolved in the run log to MuSGD |
| Initial optimizer learning rate | 0.01 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Training seed | 0 |
| Deterministic mode | true |
| AMP | enabled |
| Early stopping | disabled (`patience=0`) |
| Hardware | NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB |

The best-checkpoint validation values are precision 0.80431, recall 0.71415, F1 0.75649, mAP@0.50 0.80977, and mAP@0.50:0.95 0.58741. These are provisional because validation uses pseudo-labels rather than independent human ground truth.

## 11. File and line-number index

All line numbers below refer to `Codeing/aam_ebca_fm_kwconv_main.py`.

| Component | Lines |
|---|---:|
| 80-FPS extraction | 360–428 |
| Uniform 800 selection | 431–459 |
| Ten augmentation operations | 496–545 |
| Leakage-safe grouped split | 566–613 |
| Automatic YOLO apple labels | 728–801 |
| Dataset audit | 804–827 |
| AAM | 858–874 |
| EBCA | 877–896 |
| FM | 899–919 |
| KWConv | 922–935 |
| AAM → EBCA → FM → KWConv connection | 938–952 |
| Detect-head wrapper | 955–999 |
| YOLO head insertion | 1039–1058 |
| Training function and custom trainer | 1071–1125 |
| AdaBins adapter | 1169–1205 |
| Robust instance depth | 1208–1225 |
| Apparent-scale depth correction | 1235–1272 |
| Track state | 1276–1306 |
| Depth-aware ByteTrack | 1309–1371 |
| Unique line counter | 1374–1397 |
| Detection/depth/tracking pipeline | 1400–1462 |

## 12. Audit outcome and completed code updates

- Added the requested executable main Python source exported from the complete research notebook.
- Added the exact 928×512 AdaBins input path used by the final results.
- Added the exact apparent-scale correction used by the published depth CSV, verified against the saved result values.
- Preserved the trained AAM–EBCA–FM–KWConv architecture and checkpoint compatibility.
- Documented where the PPT is conceptual and where the trained computation differs.
- Did not claim eight-head attention, positional encoding, or an EBCA-internal CSP block because those operations are not present in the trained checkpoint.
- Did not convert relative AdaBins values into claimed physical metres.

Changing EBCA to the full dense multi-head PPT architecture would constitute a new model and would require retraining and revalidation; it is not a documentation-only correction.
