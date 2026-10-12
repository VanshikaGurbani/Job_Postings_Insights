# 💼 Job Postings Insights Extraction Using NLP and ML

---

## 📚 Project Overview

This project develops a full pipeline to extract structured insights from real-world job postings data using hybrid methods — combining NLP keyword extraction, HuggingFace sentence embeddings, KMeans clustering, and exploratory visualizations.

---

## ❓ Problem Statement

Job descriptions are often messy, long, and unstructured.
Traditional methods struggle to extract meaningful fields like skills, salary, remote availability.

**Our goals:**
- Automatically extract important fields (skills, salary range, remote allowed)
- Cluster jobs based on skill similarity
- Visualize insights across the dataset

---

## 📚 Dataset Summary

- **Source:** Raw scraped US job postings (sample)
- **Size:** 123,849 job records (prototyped on sampled 200 rows)
- **Fields:** `title`, `description`, `location`, `min_salary`, `max_salary`, `pay_period`, etc.

---

## 🏠 Project Architecture

```
job_postings_insights/
├── data/                            # Raw and cleaned datasets
├── notebooks/
│   ├— 01_EDA_Cleaning.ipynb
│   ├— 02_LLM_Extraction.ipynb (attempted)
│   ├— 03_Postprocess_Extracted_Fields.ipynb
│   ├— 04A_Embeddings_Clustering.ipynb
│   └— 04B_Insights_Visualization.ipynb
├── outputs/
│   ├── extracted_data/
│   │   ├— final_structured_job_postings.csv
│   │   └— final_structured_job_postings_with_clusters.csv
│   └── plots/
│       ├— top_skills.png
│       ├— remote_vs_onsite.png
│       ├— top_locations.png
│       ├— cluster_distribution.png
│       ├— salary_by_cluster.png
│       └— remote_vs_onsite_skills.png
├── requirements.txt
├── README.md
├── LICENSE
```

---

## 🛠️ Methods Used

- **Text Cleaning:** Regular expressions, text normalization
- **Skills Extraction:** Keyword matching from descriptions
- **Remote Detection:** Keyword scanning (remote, hybrid, WFH)
- **Salary Extraction:** Building from min/max fields
- **Embeddings:** Sentence-transformers (`all-MiniLM-L6-v2`)
- **Clustering:** KMeans unsupervised clustering
- **Dimensionality Reduction:** UMAP for 2D visualization
- **Visualizations:** Top skills bar chart, salary by cluster, remote vs onsite jobs

---

## 📊 Insights and Results

- 🔹 **Top Skills:** Python, SQL, Machine Learning, AWS among most demanded
- 🔹 **Remote Jobs:** remote vs. on-site split shown in `outputs/plots/remote_vs_onsite.png` (computed on the 200-row prototype sample)
- 🔹 **Top Locations:** New York, Remote, San Francisco most popular
- 🔹 **Clusters:** Jobs grouped into skill-based clusters (tech-heavy, business, healthcare)
- 🔹 **Salary Trends:** Higher salaries linked with cloud/AI skills clusters

---

## 📸 Visualizations

📂 `/outputs/plots/` contains:
- `top_skills.png`
- `remote_vs_onsite.png`
- `top_locations.png`
- `cluster_distribution.png`
- `salary_by_cluster.png`
- `remote_vs_onsite_skills.png`

---

## ⚙️ Technologies Used

| Tool/Library | Purpose |
|:---|:---|
| Python 3 | Programming Language |
| pandas, numpy | Data wrangling |
| sentence-transformers | Embeddings generation |
| scikit-learn | Clustering (KMeans) |
| umap-learn | Dimensionality reduction |
| matplotlib, seaborn | Data visualization |
| wordcloud | Optional wordcloud generation |

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/job_postings_insights.git
   cd job_postings_insights
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebooks sequentially:
   - `01_EDA_Cleaning.ipynb`
   - `02_LLM_Extraction.ipynb` (attempted; fallback to NLP extraction)
   - `03_Postprocess_Extracted_Fields.ipynb`
   - `04A_Embeddings_Clustering.ipynb`
   - `04B_Insights_Visualization.ipynb`

---

## 🔮 Future Work

- Fine-tune NER models for even cleaner entity extraction
- Apply to full 123,849 dataset with batching
- Build Streamlit dashboard for live job insights
- Integrate more sophisticated LLMs like GPT-4/Mistral

---

## 📜 License

MIT License

---

> 📢 **Note:** Initial LLM-based extraction (using `flan-t5`) was attempted, but hybrid NLP extraction was preferred for better quality and reliability. (Real-world AI engineer mindset!)