**Dataset:** 33,211 cleaned postings (source: `hf`).

### Rules baseline coverage (no labels needed)

| Field | Value |
|---|---|
| Postings with at least one taxonomy skill | 91.6% |
| Taxonomy size | 104 skills |
| Postings with a usable annual salary | 43.8% |
| ...from the structured columns / only from the text | 13,332 / 1,211 |
| Work arrangement (remote / hybrid / onsite / unknown) | 11.7% / 9.3% / 13.7% / 65.3% |
| Postings the original substring rule tags with skill "R" vs the regex rule | 100.0% vs 0.5% |
| "Remote" share: original rule (any mention, incl. hybrid) vs regex remote+hybrid | 22.9% vs 21.0% |

### Extraction accuracy vs hand labels

_Not available: the hand-labelled set has not been scored yet._

### Skill clusters

20,928 of 33,211 postings have at least 2 non-soft skills and are clustered (hard-skill binary x IDF -> SVD(15) -> L2).

| Metric | Value |
|---|---|
| k chosen from sweep (silhouette among seed-stable k) | 10 |
| Silhouette (cosine) at chosen k | 0.252 |
| Seed stability, mean pairwise ARI over 5 seeds | 0.709 |
| Original method (raw binary skills, k=15), silhouette | 0.088 |

| Cluster | Size | Top skills | Titles in data/software/AI role families |
|---|---|---|---|
| 0: Sales / CRM / Marketing | 1,833 | Sales, CRM, Marketing, Customer Service | 1.4% |
| 1: Recruiting / Compliance / Customer Service | 1,595 | Recruiting, Compliance, Customer Service | 3.6% |
| 2: Marketing / Customer Service | 1,833 | Marketing, Customer Service | 5.0% |
| 3: Python / SQL / AWS | 2,864 | Python, SQL, AWS, Java | 52.9% |
| 4: Accounting / Financial Analysis / Auditing | 2,342 | Accounting, Financial Analysis, Auditing, Compliance | 2.6% |
| 5: Excel / Microsoft Office / Compliance | 2,574 | Excel, Microsoft Office, Compliance | 3.3% |
| 6: Customer Service / Compliance | 2,051 | Customer Service, Compliance | 1.4% |
| 7: Patient Care / CPR/BLS / Customer Service | 1,758 | Patient Care, CPR/BLS, Customer Service, Compliance | 0.1% |
| 8: Project Management / Budgeting / Compliance | 2,100 | Project Management, Budgeting, Compliance | 4.4% |
| 9: Supply Chain / Microsoft Office / Excel | 1,978 | Supply Chain, Microsoft Office, Excel, Compliance | 3.0% |

### Salary regression (postings that state pay)

14,480 of 33,211 postings (43.6%) state a usable salary. 5-fold GroupKFold grouped by normalised title, out-of-fold predictions. Target: log annual mid-point (USD); hourly x2080, weekly x52, monthly x12.

| Model | MAE (USD) | Median AE (USD) | MAPE | R^2 (log) |
|---|---|---|---|---|
| Global median (baseline) | 43,480 | 35,000 | 52.3% | -0.00 |
| Median by role family x experience (baseline) | 36,065 | 25,000 | 41.6% | 0.23 |
| Ridge (title TF-IDF + skills + categories) | 26,725 | 16,712 | 28.8% | 0.58 |
| Gradient boosting (same features) | 26,944 | 16,850 | 28.8% | 0.57 |
