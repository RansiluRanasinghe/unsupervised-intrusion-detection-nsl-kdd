# 🔐 Unsupervised Network Intrusion Detection using K-Means

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-Educational-blue?style=flat-square)

### Dataset: NSL-KDD

This project implements an **unsupervised machine learning approach** to detect anomalous network traffic using **K-Means clustering**. The goal is to identify **intrusion patterns without using labeled data during training**, simulating real-world intrusion detection scenarios where labeled attack data may be unavailable or incomplete.

The project uses the **NSL-KDD dataset**, a refined and widely accepted benchmark dataset for network intrusion detection research.

---

## 🎯 Objectives

- ✅ Detect anomalous network traffic using **unsupervised learning**
- ✅ Cluster network connections into **normal and attack behavior**
- ✅ Validate clustering results using ground-truth labels (evaluation only)
- ✅ Demonstrate a **clean and realistic ML pipeline** suitable for academic coursework

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Dataset** | NSL-KDD (Improved version of KDD Cup 99) |
| **Source** | Kaggle |
| **Records** | Network connection logs |
| **Features** | 41 traffic-related attributes |
| **Labels** | Attack types (used only for evaluation) |
| **Data Types** | Numerical + Categorical |

### Files Used
- `KDDTrain+.txt` – Training data
- `KDDTest+.txt` – Evaluation data (recommended improvement)

---

## 🧠 Methodology

### 1️⃣ Data Loading
- Loaded raw `.txt` files without headers
- Assigned feature indices programmatically

### 2️⃣ Data Preprocessing
- Separated features and labels
- Dropped difficulty-level column
- **One-hot encoded** categorical features
- **Standardized** numerical features using `StandardScaler`

### 3️⃣ Model Training
- **Algorithm**: K-Means Clustering
- **Number of clusters**: K = 2
- **Training**: Performed **without labels** (fully unsupervised)

### 4️⃣ Model Evaluation
- Cluster-to-label mapping using **majority voting**
- **Evaluation metrics**:
  - Cluster purity
  - Accuracy
  - Precision
  - Recall
- Analysis of false positives and false negatives

---

## 📈 Key Results

- ✅ Successfully separated **normal vs anomalous traffic patterns**
- ✅ Detected multiple attack types **without prior label information**
- ✅ Demonstrated realistic limitations of distance-based clustering models

---

## 🛠️ Technologies Used

### Machine Learning
- **Python** — Programming language
- **scikit-learn** — K-Means implementation, preprocessing, evaluation
- **NumPy** — Numerical computation
- **Pandas** — Data manipulation

### Development
- **Jupyter Notebook / Google Colab** — Interactive development environment

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- pip or conda

### Installation
```bash
# Clone the repository
git clone https://github.com/RansiluRanasinghe/Network-Intrusion-Detection-KMeans.git
cd Network-Intrusion-Detection-KMeans

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Run the Notebook
Open and run `intrusion_detection.ipynb` sequentially.

---

## ⚠️ Limitations

- ❌ K-Means assumes **spherical clusters**
- ❌ **High dimensionality** after one-hot encoding
- ❌ Overlap between normal and attack traffic patterns
- ❌ Not optimized for **real-time deployment**

---

## 🔮 Future Improvements

- [ ] Train on `KDDTrain+` and evaluate on `KDDTest+` for proper generalization
- [ ] Apply **feature selection** to reduce dimensionality
- [ ] Add clustering metrics (**Silhouette Score**, **Davies-Bouldin Index**)
- [ ] Visualize clusters using **PCA** or **t-SNE**
- [ ] Compare with **Isolation Forest** or **DBSCAN**
- [ ] Use **autoencoders** for deep anomaly detection
- [ ] Implement **real-time detection** pipeline

---

## 📌 Key Learning Outcomes

This project demonstrates:

✔ **Unsupervised learning** for cybersecurity  
✔ **Real-world data preprocessing** for network traffic  
✔ **Cluster evaluation** without relying on labeled training  
✔ **Ethical intrusion detection** system design  
✔ **Academic-grade methodology** and documentation

### Skills Demonstrated
- Unsupervised machine learning
- Network traffic analysis
- High-dimensional data preprocessing
- Cybersecurity applications
- Model evaluation with pseudo-labels

---

## 👨‍🎓 Academic Context

This project was developed as part of a **Machine Learning coursework module**, focusing on:
- Unsupervised learning techniques
- Real-world data preprocessing challenges
- Ethical and practical intrusion detection systems

### Important Note
> **Labels were never used during training** — only for post-clustering evaluation.  
> This simulates realistic intrusion detection scenarios where labeled attack data is limited.

---

## 🎯 Use Cases

This approach can be adapted for:
- **Cybersecurity** — Network intrusion detection systems (NIDS)
- **Fraud Detection** — Identifying anomalous transactions
- **Industrial IoT** — Detecting abnormal sensor behavior
- **Cloud Security** — Monitoring unusual access patterns
- **Research** — Benchmarking unsupervised anomaly detection

---

## ✅ Conclusion

This project demonstrates that **unsupervised learning techniques** can effectively identify abnormal network behavior, making them suitable for **real-world cybersecurity applications** where labeled data is limited or unavailable.

The approach highlights both the **strengths** (label-free learning) and **limitations** (cluster overlap) of distance-based methods for intrusion detection.

---

## 📜 License

This project is intended for **educational and research purposes only**.

---

## 🙏 Acknowledgements

- **NSL-KDD Dataset** — University of New Brunswick
- **Kaggle** — Dataset hosting platform

### Dataset Citation
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set.

---

## 🤝 Connect

**Ransilu Ranasinghe**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ransilu-ranasinghe-a596792ba)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/RansiluRanasinghe)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:dinisthar@gmail.com)

**Interests:**  
Machine Learning • Cybersecurity • Anomaly Detection • Unsupervised Learning

---

<div align="center">

**⭐ If you find this project useful, consider giving it a star!**

**Built for cybersecurity awareness and academic excellence.**

</div>
