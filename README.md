# Video-Compression-using-ML
# Video Compression Using Unsupervised Learning

### Group 24:

## 📌 Introduction

This project demonstrates a custom video compression pipeline using:
- **Block-based motion compensation**
- **K-means clustering**
- **Residual frame encoding**

The objective is to reduce the video file size while maintaining visual quality, evaluated using **Peak Signal-to-Noise Ratio (PSNR)**. The techniques used are similar in spirit to modern video codecs like H.264 and HEVC.

---

## 🛠️ Key Techniques and Workflow

### 1. Video Initialization
- Video is loaded using `cv2.VideoCapture()`.
- Block size: `16x16`
- Number of clusters for K-means: `5`
- Only the first 60 seconds (≈1440 frames at 24 fps) are processed.

### 2. Frame Division
- Each frame is split into 16×16 pixel blocks.
- This granularity simplifies motion tracking and compression.

### 3. K-means Clustering
- Each block's average color is computed.
- K-means is applied to reduce frame complexity by grouping similar blocks.

### 4. Motion Compensation
- **Diamond Search Algorithm** used for block-based motion vector calculation between consecutive frames.
- Captures motion to allow inter-frame compression.

### 5. Compression Pipeline
For each frame, the following components are saved:
- Cluster assignments (block-wise)
- Motion vectors (motion model)
- Residual frame (difference between original and predicted)
- Frame metadata (shape, block size)

### 6. Decompression
- Motion vectors are applied to predict frame content.
- Residual frame is added to recover the original frame.

### 7. Refresh Mechanism
- Reference frames are updated every 4 frames to limit error propagation and enhance video quality.

---

## 📈 Performance Evaluation

- **Metric:** PSNR (Peak Signal-to-Noise Ratio)
- PSNR is computed for each reconstructed frame.
- A higher PSNR value indicates better visual fidelity.

---

## 📊 Results

- The method successfully compresses video while maintaining good playback quality.
- Average PSNR across frames indicates satisfactory performance for practical use cases such as streaming or storage.

---

## 🔚 Conclusion

This project showcases a foundational video compression method combining unsupervised learning and motion estimation. Future improvements could include:
- Experimenting with alternative clustering techniques
- Adaptive block sizes
- Advanced motion compensation models

---

## 📚 References

1. Kaur, A., and R. Kaur. "Feature extraction from video data..." IRJET, 2015.
2. Pfaff, J., et al. "Video compression using generalized binary partitioning..." IEEE TCSVT, 2019.
3. Maksimov, A., and Gashnikov, M. "Generalization of ML-Based Image Compression..." ITNT 2023.
4. Tang, H., et al. "Deep unsupervised key frame extraction..." ACM TOMM, 2023.
5. Lampert, C. H. "Machine learning for video compression..." ICPR, 2006.

---

## 💻 Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-learn (for K-means)

---


