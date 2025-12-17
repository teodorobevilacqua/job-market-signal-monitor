# Job Market Signal Monitor

Applied machine learning project for extracting **interpretable job market signals** from unstructured job postings using semantic NLP and unsupervised learning.

---

## 🔍 Problem
Job postings contain rich information about labor demand, but:
- job titles are noisy and heterogeneous,
- keyword-based analysis misses semantic structure,
- short-term dynamics are hard to interpret.

This project builds a **monitoring pipeline** that groups postings into coherent job families and tracks **short-window shifts in their relative prominence**.

---

## 📊 Data
- Source: Online job postings dataset (titles, descriptions, metadata)
- Time coverage: ~2 months  
- Scope: **Cross-sectional, short-window monitoring**

> Results are interpreted as **relative composition shifts**, not long-term labor market trends.

---

## 🧠 Approach

**1. Data preparation**
- Cleaning and standardization
- Efficient storage (Parquet)
- Reproducible project structure

**2. Semantic embeddings**
- Job descriptions encoded using **Sentence-BERT**
- Dense vector representation captures meaning beyond keywords

**3. Clustering**
- KMeans applied to embedding space
- Cluster count selected via elbow method
- Full-dataset clustering with stability considerations

**4. Human-in-the-loop labeling**
- Automatic cluster labels reviewed and refined
- Ambiguous job families explicitly acknowledged
- Focus on interpretability and realism

**5. Signal extraction**
- Job family shares compared between early vs late periods
- Relative changes (Δ share) used as monitoring signals

---

## 📈 Key Results (Short-Window Signals)

**Rising job families**
- Software Engineering
- Electrical & Mechanical Engineering
- Project Management
- Network / Technical Support
- Outside Sales

**Declining job families**
- RN / CNA-related roles
- RN–LTAC / Nurse Practitioner roles
- Sales Management / Business Development
- Retail supervision and customer service roles

---

## ✅ Why This Matters
- Demonstrates how **semantic embeddings + unsupervised learning** reveal occupational structure
- Produces **interpretable, job-family–level signals**
- Works even with limited temporal coverage
- Mirrors real-world hiring analytics and monitoring systems

---

## 🗂️ Project Structure
notebooks/      # Exploratory analysis
src/            # Reusable pipeline components
models/         # Clustering outputs
docs/           # Figures and summaries
README.md
requirements.txt

## ⚠️ Limitations
- Short observation window
- Results depend on embedding model and K selection
- Signals reflect **relative composition**, not absolute employment

These constraints are explicit and aligned with the project’s monitoring goal.

---

## 🚀 Next Steps
- Extend to longer rolling windows
- Track cluster stability over time
- Extract skill-level signals within clusters
- Deploy as a lightweight monitoring service

---

## 👤 Author
**Applied Machine Learning Engineer**  
Focus: NLP, unsupervised learning, interpretable monitoring systems# job-market-signal-monitor
