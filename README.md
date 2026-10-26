# Job Postings Insights

A small NLP / analytics pipeline over US job postings. It extracts **skills, salary, work arrangement
(remote / hybrid / onsite)** and **work-authorisation signals for F-1 OPT / STEM OPT candidates**
(sponsorship offered, no sponsorship, OPT welcomed, contract / staffing-style postings), scores each method
against labelled examples, clusters postings by skills, models salary, and serves a dashboard.

![Dashboard](docs/dashboard.png)

## What it does

| Piece | What you get |
|---|---|
| Extraction | Regex skills taxonomy (~100 skills), salary parser (text + structured columns, annualised to USD), remote/hybrid/onsite rules, work-authorisation rules (`src/jobinsights/`) |
| Evaluation | Precision / recall / F1 with bootstrap intervals for the old substring matcher, the regex rules, and a free local LLM (`Qwen2.5-1.5B-Instruct`, CPU) |
| Clusters | Skill-based job clusters, k chosen from silhouette and seed stability, clusters named by distinctive skills |
| Salary model | Ridge / gradient boosting vs median baselines, out-of-fold errors, folds grouped by title |
| Dashboard | Skills demand, remote share by role, salary by role and state, cluster map; filters for work-authorisation, contract/staffing and OPT mentions |

## Results

<!-- RESULTS:START -->
**Dataset:** 33,211 cleaned postings (source: `hf`).

### Rules baseline coverage (no labels needed)

| Field | Value |
|---|---|
| Postings with at least one taxonomy skill | 91.6% |
| Taxonomy size | 104 skills |
| Postings with a usable annual salary | 43.8% |
| ...from the structured columns / only from the text | 13,332 / 1,211 |
| Work arrangement (remote / hybrid / onsite / unknown) | 12.4% / 10.3% / 12.0% / 65.4% |
| Work authorisation (friendly / restricted / mixed / not stated) | 0.5% / 4.3% / 0.4% / 94.7% |
| Postings mentioning OPT / STEM OPT / F-1 | 0.4% |
| Contract / staffing-style postings | 24.6% |
| Postings the original substring rule tags with skill "R" vs the regex rule | 100.0% vs 0.5% |
| "Remote" share: original rule (any mention, incl. hybrid) vs regex remote+hybrid | 22.9% vs 22.6% |

### Extraction accuracy vs reference labels

30 postings with reference labels (stratified, not population-proportional; 95% bootstrap CI on F1 in brackets). The rules were corrected after a first scoring pass on these same postings, so their scores are optimistic; the local LLM was not tuned.

| System | Task | Precision | Recall | F1 |
|---|---|---|---|---|
| legacy | skills_18_legacy_vocab | 0.41 | 0.89 | 0.56 (0.46-0.64) |
| legacy | offers_remote_or_hybrid | 0.90 | 0.82 | 0.86 (0.67-1.00) |
| rules | skills_full_taxonomy | 0.74 | 0.75 | 0.75 (0.66-0.81) |
| rules | skills_18_legacy_vocab | 0.81 | 0.67 | 0.73 (0.62-0.82) |
| rules | salary_presence | 1.00 | 0.88 | 0.93 (0.75-1.00) |
| rules | salary_strict | 0.71 | 0.62 | 0.67 (0.29-1.00) |
| rules | offers_remote_or_hybrid | 0.92 | 1.00 | 0.96 (0.84-1.00) |
| llm:Qwen2.5-1.5B-Instruct | skills_full_taxonomy | 0.74 | 0.23 | 0.35 (0.23-0.46) |
| llm:Qwen2.5-1.5B-Instruct | skills_18_legacy_vocab | 0.73 | 0.18 | 0.29 (0.12-0.44) |
| llm:Qwen2.5-1.5B-Instruct | salary_presence | 0.30 | 0.88 | 0.45 (0.21-0.67) |
| llm:Qwen2.5-1.5B-Instruct | salary_strict | 0.30 | 0.88 | 0.45 (0.21-0.67) |
| llm:Qwen2.5-1.5B-Instruct | offers_remote_or_hybrid | 1.00 | 0.64 | 0.78 (0.50-0.95) |

### Work-authorisation and contract signals (rules vs reference labels)

45 labelled postings, including ones the rule flagged (friendly / restricted) and random ones. Flagged postings are over-represented, so **only precision is meaningful here; recall cannot be estimated** because only about 5% of postings say anything about work authorisation.

| Signal | Precision | n labelled |
|---|---|---|
| Rule says "friendly" (sponsorship / OPT welcomed) | 1.00 | 45 |
| Rule says "restricted" or "mixed" | 0.93 | 45 |
| Rule says contract / staffing-style | 0.39 | 45 |

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
| Ridge (title TF-IDF + skills + categories) | 26,720 | 16,728 | 28.8% | 0.58 |
| Gradient boosting (same features) | 26,885 | 16,669 | 28.8% | 0.57 |

<!-- RESULTS:END -->

## Run it

```bash
pip install -e ".[dev,dashboard,llm]"
jobinsights download --source hf     # public 33k-row subset, no login (or --source kaggle / local)
jobinsights run-all                  # clean -> extract -> clusters -> salary model -> report
jobinsights dashboard                # Streamlit dashboard
pytest && ruff check .               # tests and lint (also run in CI)
```

Optional, for the extraction scores: `jobinsights make-eval-set`, `jobinsights llm-extract --labelled`,
`jobinsights label` (labelling app), `jobinsights evaluate`.

## Data

The full dataset is the LinkedIn Job Postings (2023-2024) table on Kaggle
([`arshkon/linkedin-job-postings`](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings),
123,849 rows; Kaggle needs a login, so use `--source kaggle` with your own credentials or `--source local`).
The numbers above come from the public Hugging Face mirror
[`xanderios/linkedin-job-postings`](https://huggingface.co/datasets/xanderios/linkedin-job-postings)
(33,246 rows, MIT), which needs no login. The data covers all kinds of jobs, not only data roles.

## Limitations

* The reference set is small (see n in the table), stratified rather than representative, and produced in a
  single annotation pass, so confidence intervals are wide and there is no agreement estimate.
* The rules were written and corrected while looking at this dataset and at the reference set, so their
  scores are optimistic. The local LLM is a 1.5B-parameter model on CPU; it says nothing about larger models.
* Only about 5% of postings say anything about work authorisation, so the OPT filter returns few postings;
  recall for that signal is not estimable. It is a text signal, not a list of H-1B sponsors (the public subset
  has no company names; the full Kaggle file does, which would allow a join with public sponsor data).
  The contract/staffing flag is weak (precision about 0.4).
* Salary is modelled only for postings that state pay (about 44%); they are not a random sample and pay is
  not imputed. Hourly pay is annualised at 2,080 hours.
* Cluster silhouette scores for the old and new methods are in different spaces and not strictly comparable.

## Layout

```
src/jobinsights/   data, taxonomy, remote, salary, visa, extract, llm, evaluate, cluster, salary_model, report, apps, cli
tests/             pytest for parsers, normalisers, metrics, clustering helpers
docs/              UPGRADE_PLAN.md (audit of the original prototype), dashboard.png
outputs/           RESULTS.md and JSON results written by the pipeline
notebooks/legacy/  the original prototype notebooks and plots (not reproducible, kept for history)
```
