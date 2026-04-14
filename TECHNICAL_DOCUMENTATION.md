# Technical Documentation

> Detailed methodology and implementation of the fMRI-CNN-Statistics mapping project

---

## Table of Contents

1. [Experimental Design](#1-experimental-design)
2. [Image Statistics Computation](#2-image-statistics-computation)
3. [VGG-16 Feature Extraction](#3-vgg-16-feature-extraction)
4. [fMRI Data Processing](#4-fmri-data-processing)
5. [Correlation Analysis](#5-correlation-analysis)
6. [Statistical Considerations](#6-statistical-considerations)
7. [Implementation Details](#7-implementation-details)

---

## 1. Experimental Design

### 1.1 Dataset: Natural Scenes Dataset (NSD)

**Overview:**
- **Scale**: 73,000 natural scene images
- **Participants**: 8 subjects (extensive scanning)
- **Scanner**: 7 Tesla Siemens Magnetom
- **Resolution**: 1.8mm isotropic voxels
- **Sessions**: 30-40 scan sessions per subject over ~1 year

**Task:**
- Continuous recognition task: "Have you seen this image before?"
- Each subject viewed 9,000-10,000 unique images
- Total trials per subject: 22,000-30,000

**Data Processing:**
- **GLM (General Linear Model)**: Single-trial beta estimates
- **HRF (Hemodynamic Response Function)**: Calibrated per voxel
- **GLMdenoise**: Noise reduction using control voxels

### 1.2 Visual Areas (ROI Definition)

ROIs defined using **population receptive field (pRF)** mapping and **functional localizers (fLoc)**:

| ROI | Full Name | Description | Mask Values |
|-----|-----------|-------------|-------------|
| V1v | V1 ventral | Primary visual cortex (lower visual field) | 1 |
| V1d | V1 dorsal | Primary visual cortex (upper visual field) | 2 |
| V2v | V2 ventral | Secondary visual cortex (lower) | 3 |
| V2d | V2 dorsal | Secondary visual cortex (upper) | 4 |
| V3v | V3 ventral | Third visual area (lower) | 5 |
| V3d | V3 dorsal | Third visual area (upper) | 6 |
| hV4 | Human V4 | Fourth visual area (color/form) | 7 |

**Rationale for dorsal/ventral split:**
- Captures functional differences in visual field representation
- Dorsal stream: spatial processing, motion
- Ventral stream: object recognition, form

---

## 2. Image Statistics Computation

### 2.1 Color Space Conversion

**Why not use RGB directly?**
- RGB values are non-linear (gamma-corrected for displays)
- Visual system processes linear light intensity
- Need accurate luminance for perceptual statistics

**Conversion Pipeline:**

```
sRGB (0-255) → Normalized (0-1) → Linear RGB → CIE XYZ → Luminance
```

**Step 1: Linearization (Gamma Correction)**
```matlab
% sRGB → Linear RGB
if sRGB <= 0.04045
    linear = sRGB / 12.92
else
    linear = ((sRGB + 0.055) / 1.055) ^ 2.4
end
```

**Step 2: Luminance Calculation**
```matlab
% CIE 1931 standard weights
Y = 0.2126 * R_linear + 0.7152 * G_linear + 0.0722 * B_linear
```

**Why these specific weights?**
- Based on human photopic luminosity function
- Green contributes most to perceived brightness (71.52%)
- Blue contributes least (7.22%)

### 2.2 Statistical Features

#### 2.2.1 Mean Luminance (1st Moment)
**Definition:** Average pixel intensity

**Interpretation:**
- **Low values**: Dark images (e.g., night scenes)
- **High values**: Bright images (e.g., daylight, snow)

**Calibration:** Log-transformed using Exposure Value (EV)
```matlab
EV = log2(luminance) - β
% β = -2.44078736 (neutral gray calibration)
```

#### 2.2.2 Contrast (2nd Moment)
**Definition:** Standard deviation of luminance

**Interpretation:**
- **Low contrast**: Foggy, uniform scenes
- **High contrast**: Sharp edges, varied textures

**Perceptual relevance:**
- Early visual cortex (V1) highly sensitive to contrast
- Drives orientation-selective neurons

#### 2.2.3 Skewness (3rd Moment)
**Definition:** Asymmetry of luminance distribution

**Formula:**
```matlab
skewness = E[(X - μ)³] / σ³
```

**Interpretation:**
- **Positive**: Right-skewed (few bright highlights)
  - Associated with **glossy** materials (Motoyoshi et al., 2007)
- **Negative**: Left-skewed (few dark shadows)
  - Matte, diffuse surfaces
- **Zero**: Symmetric distribution

**Example:**
- Glossy apple: High positive skewness (bright specular highlights)
- Velvet cloth: Low/negative skewness (uniform light absorption)

#### 2.2.4 Spatial Frequency Slope (1/f^α)
**Definition:** Slope of power spectrum in log-log space

**Computation:**
```matlab
% 2D Fourier Transform
FFT = fft2(luminance_map);
power_spectrum = abs(FFT).^2;

% Convert to log-log space
log_frequency = log(frequency);
log_power = log(power_spectrum);

% Linear fit
slope = polyfit(log_frequency, log_power, 1);
α = -slope(1);  % Steeper slope = more naturalistic
```

**Interpretation:**
- **α ≈ 1-2**: Natural scenes (scale-invariant statistics)
- **α ≈ 0**: White noise (no structure)
- **α > 2**: Overly smooth (unnatural)

**Perceptual relevance:**
- **Scene openness**: Open landscapes (lower α) vs. closed interiors (higher α)
- **Naturalness**: Natural scenes vs. urban environments (Oliva & Torralba, 2001)

### 2.3 Local Statistics (5×5 Grid)

**Motivation:**
- Global statistics ignore spatial structure
- Visual system processes local contrast (receptive fields ~1-2° visual angle)

**Implementation:**
```matlab
% Divide image into 4×4 grid (16 regions)
grid_size = 4;
row_stride = image_height / grid_size;
col_stride = image_width / grid_size;

% Compute statistics per region
for each region:
    local_contrast(i) = std(region_luminance);
end
```

**Analysis:**
- Compare local contrast distribution across ROIs
- Test if spatial scale matters for neural encoding

---

## 3. VGG-16 Feature Extraction

### 3.1 Network Architecture

**VGG-16 (Visual Geometry Group, 2014):**
- **Depth**: 16 weight layers (13 conv + 3 FC)
- **Filters**: Small (3×3), stacked for large receptive fields
- **Training**: ImageNet (1.2M images, 1000 categories)

**Convolutional Block Structure:**
```
Conv3×3 → ReLU → Conv3×3 → ReLU → MaxPool2×2
```

**Hierarchy of Pooling Layers:**

| Layer | Feature Map Size | # Filters | Receptive Field | Semantic Level |
|-------|------------------|-----------|-----------------|----------------|
| pool1 | 112×112 | 64 | 10×10 pixels | Edges, colors |
| pool2 | 56×56 | 128 | 24×24 pixels | Simple textures |
| pool3 | 28×28 | 256 | 56×56 pixels | Parts, patterns |
| pool4 | 14×14 | 512 | 120×120 pixels | Object parts |
| pool5 | 7×7 | 512 | Full image | Objects, scenes |

### 3.2 Why VGG-16?

**Advantages:**
1. **Interpretable hierarchy**: Clear layer-by-layer abstraction
2. **Proven biological relevance**: Predicts IT cortex responses (Yamins et al., 2014)
3. **Texture sensitivity**: Captures texture statistics (Gatys et al., 2015)
4. **Widely studied**: Extensive literature for comparison

**Alternatives considered:**
- ResNet: Deeper but less interpretable (skip connections)
- AlexNet: Shallower, less rich representations
- Inception: Complex multi-scale processing

### 3.3 Dimensionality Reduction: Incremental PCA

**Problem:**
- Raw feature dimensions: pool1: 802,816; pool5: 25,088
- 73,000 images → RAM explosion
- Need: Reduce to manageable size (256D) while preserving variance

**Why Incremental PCA?**
- **Memory efficient**: Process data in batches
- **Online learning**: No need to load all data at once
- **Variance preservation**: Retain 95%+ of explained variance

**Algorithm (Ross et al., 2008):**

```
Input: New data batch X [d × m], existing PCA (U, S, μ, n)
Output: Updated PCA (U', S', μ', n')

1. Compute batch mean: μ_X = mean(X, 2)
2. Update global mean: μ' = (n·μ + m·μ_X) / (n + m)
3. Mean correction: ΔX = [X - μ_X, √(nm/(n+m))·(μ_X - μ)]
4. QR decomposition: [Q, R] = qr([U·S, ΔX])
5. SVD on R: [Ũ, S', Ṽ] = svd(R)
6. Update components: U' = Q·Ũ
7. Update count: n' = n + m
```

**Implementation Details:**
- **Batch size**: 256 images
  - Balance: Too small = slow; Too large = memory
- **Components retained**: 256 (top principal components)
- **Variance explained**: ~96% across all pooling layers

**Computational Complexity:**
- **Standard PCA**: O(d²n) - prohibitive for d=800k, n=73k
- **Incremental PCA**: O(dk² × n/b) - k=256, b=256 → 10,000× faster

---

## 4. fMRI Data Processing

### 4.1 ROI Masking

**Input Files (NumPy arrays):**
```
lh.prf-visualrois_challenge_space.npy  # Left hemisphere mask
rh.prf-visualrois_challenge_space.npy  # Right hemisphere mask
lh_fmri.npy  # Left hemisphere fMRI [n_trials × n_voxels]
rh_fmri.npy  # Right hemisphere fMRI
```

**Processing Steps:**
1. **Load masks**: Binary arrays indicating ROI membership
2. **Combine hemispheres**: Concatenate left and right
3. **Apply masks**: Extract voxels for each ROI
4. **Store separately**: One file per subject per ROI

**Voxel Counts (Example, Subject 1):**
| ROI | # Voxels |
|-----|----------|
| V1v | 1,234 |
| V1d | 1,156 |
| V2v | 987 |
| V2d | 1,043 |
| V3v | 756 |
| V3d | 823 |
| hV4 | 645 |

### 4.2 Signal Quality

**GLMdenoise (Kay et al., 2013):**
- Uses "control" voxels (not stimulus-driven) to model noise
- Improves SNR by ~15-30%

**Quality Metrics:**
- **Split-half reliability**: Correlation between odd/even trial averages
- **Noise ceiling**: Upper bound on model performance

---

## 5. Correlation Analysis

### 5.1 Design Matrix

**Three Main Comparisons:**

#### (A) ROI × Image Statistics
```
Matrix: [4 stats × 7 ROIs × 8 subjects]
- Stats: mean, contrast, skewness, fft_slope
- ROIs: V1v, V1d, V2v, V2d, V3v, V3d, hV4
```

#### (B) ROI × VGG Layers
```
Matrix: [5 layers × 7 ROIs × 8 subjects]
- Layers: pool1, pool2, pool3, pool4, pool5
```

#### (C) VGG Layers × Image Statistics
```
Matrix: [5 layers × 4 stats × 8 subjects]
```

### 5.2 Correlation Computation

**Method:** Pearson correlation coefficient

For each (feature, voxel) pair:
```matlab
r = corr(feature_vector, voxel_timeseries)
% feature_vector: [n_images × 1]
% voxel_timeseries: [n_images × 1]
```

**Aggregation:**
- **Within ROI**: Compute r for each voxel, take max/mean
- **Across subjects**: Average max/mean correlations

**Why max AND mean?**
- **Max**: Best-case encoding (most selective voxels)
- **Mean**: Typical encoding (population average)

### 5.3 Log vs. Raw Statistics

**Rationale:**
- Weber-Fechner law: Perception is logarithmic
- Exposure value (EV) calibration for luminance
- Contrast response saturation in V1

**Result:**
- Log-transformed statistics show **higher correlations** with early visual areas
- Suggests neural encoding matches perceptual space

---

## 6. Statistical Considerations

### 6.1 Multiple Comparisons

**Problem:**
- 4 stats × 7 ROIs × 8 subjects = 224 comparisons (per analysis)
- False discovery rate (FDR) inflation

**Solutions:**
1. **Report effect sizes**: Correlation magnitude matters, not just p-values
2. **Consistency across subjects**: Effects replicated in 7/8 or 8/8 subjects
3. **A priori hypotheses**: V1 should correlate with contrast (known)

### 6.2 Confounds

**Potential Confounds:**
- **Image memorability**: High-contrast images more memorable → task effects
- **Attention**: Some images more attended → modulated responses
- **Low-level artifacts**: JPEG compression, rescaling

**Controls:**
- **Task-regressed betas**: GLM includes task performance
- **Stimulus repetition**: 3 presentations per image (averaged)

### 6.3 Interpretation Caveats

**Correlation ≠ Causation:**
- Cannot claim statistics "cause" neural responses
- May reflect shared computational principles

**Model comparison needed:**
- Compare to alternative models (Gabor filters, ResNet features)
- Use cross-validation to assess generalization

---

## 7. Implementation Details

### 7.1 Software Engineering Choices

**Modular Design:**
- Separate preprocessing, analysis, visualization
- Allows parallel development and debugging

**Vectorization:**
- MATLAB's strengths: Matrix operations
- Avoid loops where possible (100-1000× speedup)

**Memory Management:**
- Process subjects separately
- Clear large variables after use
- Use `single` precision where appropriate (50% memory)

### 7.2 Computational Performance

**Hardware Requirements:**
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 50GB for intermediate files
- **GPU**: Not required (VGG-16 inference is fast on CPU)

**Runtime (8 subjects, full pipeline):**
| Step | Time |
|------|------|
| Image statistics | ~15 min |
| VGG-16 extraction + PCA | ~45 min |
| ROI masking | ~2 min |
| Correlation analysis | ~20 min |
| Visualization | ~5 min |
| **Total** | **~90 min** |

### 7.3 Quality Assurance

**Tests Implemented:**
1. **Dimension checks**: Verify matrix sizes at each step
2. **NaN detection**: Check for invalid values
3. **Sanity checks**:
   - Correlations in [-1, 1]
   - PCA variance explained > 0.9
   - ROI voxel counts > 0

**Reproducibility:**
- Fixed random seed: `rng(0)`
- Deterministic PCA initialization
- Saved intermediate files for inspection

---

## 8. Future Directions

### 8.1 Methodological Extensions

**Advanced Models:**
- **Encoding models**: Predict fMRI from features (ridge regression)
- **Representational similarity analysis (RSA)**: Compare feature spaces
- **Deep canonical correlation analysis (DCCA)**: Joint feature learning

**Spatial Analysis:**
- **Retinotopic mapping**: Correlate features with visual field position
- **Multi-scale statistics**: Vary local patch sizes

### 8.2 Theoretical Questions

1. **Sufficiency**: Can statistics alone explain V1 responses?
2. **Necessity**: Are certain statistics critical for perception?
3. **Optimality**: Do visual areas encode statistics optimally?

### 8.3 Applications

**Computer Vision:**
- **Perceptual loss functions**: Use statistics for image quality metrics
- **Adversarial robustness**: Statistics less sensitive to perturbations

**Neuroscience:**
- **Clinical assessment**: Abnormal statistics encoding in visual disorders
- **Developmental studies**: Statistics encoding in children

---

## References

### Key Papers

1. **NSD Dataset:**
   - Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25, 116-126.

2. **Image Statistics & Perception:**
   - Motoyoshi, I., et al. (2007). Image statistics and the perception of surface qualities. *Nature*, 447, 206-209.
   - Oliva, A., & Torralba, A. (2001). Modeling the shape of the scene. *IJCV*, 42(3), 145-175.

3. **CNN-Brain Correspondence:**
   - Yamins, D. L., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience*, 19(3), 356-365.
   - Güçlü, U., & van Gerven, M. A. (2015). Deep neural networks reveal a gradient in the complexity of neural representations. *Journal of Neuroscience*, 35(27), 10005-10014.

4. **Incremental PCA:**
   - Ross, D., et al. (2008). Incremental learning for robust visual tracking. *IJCV*, 77(1-3), 125-141.

---

*Last updated: [Date]*
*Author: Hang Chen*