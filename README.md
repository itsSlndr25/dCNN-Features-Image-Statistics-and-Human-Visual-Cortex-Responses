# NSD MRI
# fMRI–CNN Representation Alignment (NSD Dataset)

> Built a large scale image, deep model and brain correlation analysis pipeline using 70k+ image samples and fMRI data from the Natural Scenes Dataset (NSD), deep CNN features, and image statistics.

---

## 📖 Project Overview

This project builds a full analysis pipeline to quantify representational alignment between human visual cortex (7T fMRI, NSD) and deep CNN features (VGG-16), while comparing contributions of low-level image statistics.

**Low-level image statistics** (mean, contrast, skewness, spatial frequency)

### Research Questions

- How can deep features account for the neural signals?
- What is the **relationship** between deep CNN representations and image statistics?

## 💡 Technical Highlights

- Processed 70k+ images across 8 subjects
- Extracted deep features from VGG-16 (pool1–pool5)
- Implemented incremental PCA for large-scale dimensionality reduction
- Designed correlation analysis pipeline across multimodal data (image statistics × CNN features × fMRI)
- Automated visualization & reproducible analysis workflow

---

## 🗂️ Repository Structure

```
correlation_project/
│
├── README.md                          # This file
├── TECHNICAL_DOCUMENTATION.md         # Detailed methodology
│
├── code/
│   ├── preprocessing/
│   │   ├── compute_image_statistics.m       # Calculate image statistics
│   │   └── extract_vgg16_features.m         # Extract & reduce VGG-16 features
│   │
│   ├── analysis/
│   │   ├── extract_roi_data.m               # Extract fMRI ROI data
│   │   └── compute_correlations.m           # Main correlation analysis
│   │
│   ├── visualization/
│   │   ├── visualize_correlations.m         # Generate bar charts
│   │   └── visualize_statistics.m           # Statistical distributions
│   │
│   └── utils/
│       ├── incrementalPCA.m                 # Incremental PCA algorithm
│       ├── getImageSpectSlope.m             # FFT slope calculation
│       └── readNPY.m/                       # Python numpy file reader
│
├── data/
│   └── [NSD  Data ]                 # See Data Requirements below
│   └── img_stat                     # Image statistics data
│   └── ROIs                         # Masked ROI data
│
├── results/
│   └── figures/                       # Generated visualizations
│
└── docs/
    ├── presentation_slides.pdf        # Lab meeting slides
    └── pipeline_flowchart.png         # Project summary
```

---

## 🔧 Requirements

### Software
- **MATLAB** R2020b or later
- **Deep Learning Toolbox** (for VGG-16)
- **Statistics and Machine Learning Toolbox**
- **Image Processing Toolbox**

### Data
This project uses the **Natural Scenes Dataset (NSD)**:
- **70,566** natural scene images
- **8 subjects**, 7T fMRI data
- **ROI masks** based on pRF and fLoc
- Download from: [Natural Scenes Dataset](http://naturalscenesdataset.org/)

**Expected data structure:**
```
/path/to/algonauts/data/
├── subj01/
│   ├── images/              # Training images
│   ├── fmri/               # fMRI responses (lh_fmri.npy, rh_fmri.npy)
│   └── roi_masks/          # ROI masks (*.npy)
├── subj02/
...
└── subj08/
```

---

## 🚀 Quick Start

### 1. Setup

```matlab
% Clone repository and navigate to project folder
cd correlation_project/code

% Configure data paths in each script:
% - DATA_FOLDER: path to NSD dataset
% - OUTPUT_FOLDER: path for results
```

### 2. Run Full Pipeline

Execute scripts in order:

```matlab
%% Step 1: Compute image statistics (mean, contrast, skewness, FFT slope)
run('preprocessing/compute_image_statistics.m');

%% Step 2: Extract and reduce dimensions of VGG-16 features using incremental PCA
run('preprocessing/extract_vgg16_features.m');

%% Step 3: Extract fMRI ROI data using mask (V1v, V1d, V2v, V2d, V3v, V3d, hV4)
run('analysis/extract_roi_data.m');

%% Step 4: Compute correlations between features and fMRI
run('analysis/compute_correlations.m');

%% Step 5: Generate visualizations
run('visualization/visualize_correlations.m');
run('visualization/visualize_statistics.m');
```

### 3. View Results
Generated figures will be in `results/figures/`:
- `ROIxStat_max.png` - ROI vs. statistics correlations
- `LayerxStat_max.png` - VGG layers vs. statistics
- `ROIxLayer_max.png` - ROI vs. VGG layers
- Statistical distribution histograms

---

## 📊 Analysis Pipeline

```
┌─────────────────┐
│  Natural Scene  │
│     Images      │ (73k images x 8 subjects)
└────────┬────────┘
         │
    ┌────┴─────────────────────┐
    │                           │
    ▼                           ▼
┌───────────┐            ┌──────────────┐
│  Image    │            │   VGG-16     │
│Statistics │            │  Features    │
│(Mean,     │            │ (pool1-pool5)│
│ Contrast, │            │              │
│ Skewness, │            │ + PCA (256D) │
│ FFT Slope)│            │              │
└─────┬─────┘            └──────┬───────┘
      │                         │
      └──────────┬──────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  fMRI Signal  │
         │   (7T, ROI)   │
         │ V1, V2, V3, V4│
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │  Correlation  │
         │   Analysis    │
         │ - ROI x Stat  │
         │ - ROI x Layer │
         │ - Layer x Stat│
         └───────┬───────┘
                 │
                 ▼
         ┌───────────────┐
         │Visualization  │
         │  (Bar charts, │
         │  Histograms)  │
         └───────────────┘
```

---

## 📈 Key Results Summary

### ROI × Image Statistics
- **Early visual areas (V1, V2)** show strongest correlation with **contrast** and **spatial frequency**
- **V4** shows unique sensitivity to **skewness** (related to glossiness perception)

### VGG-16 Layers × ROI
- **Pool1-Pool2** correlate strongly with **V1-V2** (early features)
- **Pool4-Pool5** show increasing correlation with **V4** (higher-level features)
- Hierarchical alignment between CNN depth and visual processing hierarchy

### Local vs. Global Statistics
- **Local contrast** (5×5 patches) provides distinct information from global contrast
- Suggests importance of **spatial scale** in visual encoding

---

## 🔬 Technical Details

### Image Statistics Computation
- **Color space conversion**: sRGB → CIE XYZ (linear space)
- **Luminance**: Weighted sum using [0.2126, 0.7152, 0.0722]
- **Log transformation**: Exposure value (EV) calibration
- **Spatial frequency**: 2D FFT slope (1/f^α natureness)

### VGG-16 Feature Extraction
- **Pre-trained** on ImageNet (1M+ images)
- **Architecture**: 5 pooling layers → hierarchical features
- **Dimensionality reduction**: 
  Implemented memory-efficient Incremental PCA to process high-dimensional feature vectors without loading full dataset into RAM.

### Correlation Analysis
- **Method**: Pearson correlation coefficient
- **Aggregation**: Max and mean across subjects (N=8)
- **Comparisons**:
  - ROI × Statistics (raw & log-transformed)
  - ROI × VGG layers
  - VGG layers × Statistics

---

## 📚 References

### Dataset
- Allen, E. J., et al. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience*, 25, 116-126.

### Deep Learning & Vision
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
- Yamins, D. L., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience*, 19(3), 356-365.

### Image Statistics
- Motoyoshi, I., et al. (2007). Image statistics and the perception of surface qualities. *Nature*, 447, 206-209.
- Oliva, A., & Torralba, A. (2001). Modeling the shape of the scene: A holistic representation of the spatial envelope. *IJCV*, 42(3), 145-175.