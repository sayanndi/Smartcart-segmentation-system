# 🧠 Customer Behaviour Segmentation using Unsupervised Learning

> Segmenting 2,240 customers into behaviour-based clusters using **K-Means** and **Agglomerative Clustering** — with Agglomerative Clustering delivering the best results.

---

## 📌 Project Overview

This project applies unsupervised machine learning to a real-world marketing dataset to uncover hidden customer behaviour patterns. By analysing spending habits, purchase channels, demographic details, and campaign responses, customers are grouped into distinct segments that businesses can use for targeted marketing, product recommendations, and retention strategies.

---

## 📂 Dataset

- **Source:** Marketing Campaign Dataset  
- **Records:** 2,240 customers  
- **Features:** 22 columns including demographics, spending across product categories, purchase channels, and campaign responses

| Column | Description |
|---|---|
| `ID` | Unique customer identifier |
| `Year_Birth` | Customer birth year |
| `Education` | Education level |
| `Marital_Status` | Marital status |
| `Income` | Annual household income |
| `Kidhome / Teenhome` | Number of children/teens at home |
| `Dt_Customer` | Date of customer enrollment |
| `Recency` | Days since last purchase |
| `MntWines / MntFruits / ...` | Amount spent per product category |
| `NumWebPurchases / NumStorePurchases / ...` | Purchases per channel |
| `NumWebVisitsMonth` | Web visits per month |
| `Complain` | Whether customer complained |
| `Response` | Response to last campaign |

---

## 🔁 Project Pipeline

```
Load Dataset → Handle Missing Values → Feature Engineering → Drop Columns
→ Outlier Visualization → Correlation Heatmap → One-Hot Encoding
→ Scaling → PCA (2D & 3D) → Find Optimal K → Clustering → Cluster Analysis
```

---

## 🔧 Steps Performed

### 1. 📥 Load Dataset
- Loaded the CSV dataset using `pandas`
- Inspected shape, data types, and null counts
- Found **24 missing values** in the `Income` column

---

### 2. 🧹 Handle Missing Values
- Imputed the 24 missing `Income` values using the **median** to avoid skew from outliers
- Verified no remaining nulls after imputation

---

### 3. ⚙️ Feature Engineering
New meaningful features were derived from existing columns:

| New Feature | Description |
|---|---|
| `Age` | `2024 - Year_Birth` |
| `TotalSpend` | Sum of all `Mnt*` columns |
| `NumChildren` | `Kidhome + Teenhome` |
| `TotalPurchases` | `NumWebPurchases + NumCatalogPurchases + NumStorePurchases` |
| `CustomerTenure` | Days since `Dt_Customer` |
| `IsParent` | Binary flag — 1 if `NumChildren > 0` |
| `SpendPerPurchase` | `TotalSpend / TotalPurchases` |
| `WebRatio` | `NumWebPurchases / TotalPurchases` |
| `DealRatio` | `NumDealsPurchases / TotalPurchases` |

---

### 4. 🗑️ Drop Columns
Dropped columns that are not useful for clustering:
- `ID` — just an identifier
- `Year_Birth` — replaced by `Age`
- `Dt_Customer` — replaced by `CustomerTenure`
- `Z_CostContact`, `Z_Revenue` — constant values (if present)

---

### 5. 📊 Visualize Outliers
- Used **boxplots** to detect outliers in `Income`, `Age`, `TotalSpend`, and purchase columns
- Applied **IQR-based filtering** to remove extreme outliers that could distort cluster boundaries

---

### 6. 🔥 Correlation Heatmap
- Generated a `seaborn` heatmap of feature correlations
- Identified strong correlations such as `Income` ↔ `TotalSpend` and `NumChildren` ↔ `DealRatio`
- Used insights to confirm feature engineering decisions

---

### 7. 🔢 One-Hot Encoding
- Applied `pd.get_dummies()` to categorical features: `Education` and `Marital_Status`
- Dropped the first dummy column to avoid multicollinearity

---

### 8. ⚖️ Feature Scaling
- Applied `StandardScaler` to normalise all features to zero mean and unit variance
- Scaling is critical for distance-based algorithms like K-Means and Agglomerative Clustering

---

### 9. 📉 PCA — Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** for visualisation and noise reduction

**2D PCA** — scatter plot of PC1 vs PC2 coloured by cluster label  
**3D PCA** — interactive 3D scatter plot using `plotly` or `matplotlib` for PC1, PC2, PC3  
- Explained variance ratio printed for each component

---

### 10. 🔍 Finding Optimal K

#### Elbow Method
- Ran K-Means for `k = 2` to `k = 10`
- Plotted **WCSS (Within-Cluster Sum of Squares)** vs number of clusters
- Identified the "elbow" at **k = 4**

#### Silhouette Score
- Computed silhouette scores for each value of k
- **k = 4** yielded the highest silhouette score, confirming the elbow result

---

### 11. 🤖 Clustering

#### K-Means Clustering
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X_scaled)
```

#### Agglomerative Clustering ✅ Best Result
```python
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=4, linkage='ward')
agglo.fit(X_scaled)
```

- **Agglomerative Clustering** with Ward linkage produced more cohesive, well-separated clusters than K-Means
- Evaluated using **Silhouette Score** and **Davies-Bouldin Index**

| Algorithm | Silhouette Score | Davies-Bouldin Index |
|---|---|---|
| K-Means | ~0.38 | ~1.12 |
| **Agglomerative (Ward)** | **~0.42** | **~0.97** |

---

### 12. 📈 Cluster Analysis & Characterisation

#### Countplot
- Plotted the **distribution of customers per cluster**
- Compared categorical features (Education, Marital Status, IsParent) across clusters using `seaborn.countplot`

#### Scatterplot
- Used **2D PCA scatter plots** coloured by cluster to visualise separation
- Plotted `Income` vs `TotalSpend` per cluster to reveal economic behaviour differences

#### Cluster Summary Table
Computed the mean of each feature per cluster:

```python
df.groupby('Cluster').mean().T
```

| Feature | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|---|---|---|---|---|
| **Label** | Budget Families | High Earners | Young Actives | Loyalists |
| Income | Low | High | Mid | Mid-High |
| TotalSpend | Low | High | Mid | Mid |
| NumChildren | High | Low | Low | Low |
| TotalPurchases | Low | High | Mid | High |
| CustomerTenure | Mid | Low | Low | High |
| DealRatio | High | Low | Mid | Low |
| WebRatio | High | Low | High | Low |

---

## 🗂️ Cluster Profiles

### 🟢 Cluster 0 — Budget Families
- Low income, high number of children
- Price-sensitive: heavy deal users, frequent web visits
- Low overall spend across all categories
- **Strategy:** Discount campaigns, family bundles

### 🔵 Cluster 1 — High Earners
- Top income bracket, no children
- Heavy spenders on wines, meat, and gold products
- Prefer catalog and store purchases
- **Strategy:** Premium offers, loyalty programs, exclusive catalogs

### 🟠 Cluster 2 — Young Actives
- Mid income, younger age group
- Online-first buyers, highly responsive to campaigns
- Moderate spending with deal usage
- **Strategy:** Digital campaigns, email marketing, flash sales

### 🟣 Cluster 3 — Loyalists
- Long customer tenure, frequent store buyers
- Consistent mid-to-high spenders, very low complaint rate
- Low recency (recently active)
- **Strategy:** Retention rewards, early access, referral programs

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.x` | Core language |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `plotly` | 3D PCA visualisation |
| `scikit-learn` | Scaling, PCA, K-Means, Agglomerative Clustering, metrics |

---



## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
jupyter
```

---

## 📊 Key Results

- **Best Algorithm:** Agglomerative Clustering (Ward linkage)
- **Optimal Clusters:** 4
- **Best Silhouette Score:** ~0.42
- **4 Distinct Segments Identified:** Budget Families, High Earners, Young Actives, Loyalists

---

## 🙌 Acknowledgements

Dataset sourced from a publicly available marketing campaign dataset commonly used for customer analytics and segmentation tasks.

---


This project is open-source and available under the [MIT License](LICENSE).
