# Job Postings Insights

Extract skills, salary, work arrangement (remote / hybrid / onsite) and **work-authorisation signals for
candidates on F-1 OPT / STEM OPT** (sponsorship offered, no sponsorship / citizen-only, OPT welcomed,
contract / staffing-style postings) from US job postings, measure how accurate each extraction method is
against hand labels, then cluster postings by skills, model salary, and explore everything in a dashboard.

![Dashboard](docs/dashboard.png)

Everything below the results table is reproducible from the command line; the table itself is regenerated
from pipeline outputs by `jobinsights run-all` and is never typed by hand.

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
| Work arrangement (remote / hybrid / onsite / unknown) | 11.7% / 9.3% / 13.7% / 65.3% |
| Work authorisation (friendly / restricted / mixed / not stated) | 0.6% / 4.1% / 0.5% / 94.8% |
| Postings mentioning OPT / STEM OPT / F-1 | 0.4% |
| Contract / staffing-style postings | 24.6% |
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

<!-- RESULTS:END -->

## Reproduce

```bash
pip install -e ".[dev,dashboard,llm]"      # add ",kaggle" to download the full dataset
jobinsights download --source hf            # public 33k-row subset, no login (or: --source kaggle / local)
jobinsights run-all                         # clean -> rules -> clusters -> salary model -> report
jobinsights llm-extract                     # free local LLM on the 150 evaluation postings (~1 h on CPU)
jobinsights make-eval-set                   # (once) draws the evaluation sample; ids are committed
jobinsights label                           # hand-label the sample in a small Streamlit app
jobinsights evaluate && jobinsights report  # precision / recall / F1 table
jobinsights dashboard                       # Streamlit dashboard
pytest && ruff check .                      # tests and lint (also run in CI)
```

Pinned versions are in `requirements.txt`. Data lives in `data/` (gitignored); only the evaluation sample's
posting ids and the labels are committed (`data/labels/`).

## Data

* **Full dataset:** the LinkedIn Job Postings (2023-2024) table on Kaggle,
  [`arshkon/linkedin-job-postings`](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings),
  `postings.csv`, 123,849 rows. Kaggle needs a login, so `jobinsights download --source kaggle` uses your own
  `~/.kaggle/kaggle.json`; or download the file yourself and run `--source local --file postings.csv`.
* **Subset used for the numbers shown above:** the public Hugging Face mirror
  [`xanderios/linkedin-job-postings`](https://huggingface.co/datasets/xanderios/linkedin-job-postings)
  (33,246 rows, MIT), which needs no login. The results block states which source produced it.
* The data covers **all kinds of jobs**, not only data roles (most postings are sales, healthcare,
  accounting, operations...). Data/ML/software roles are a small share, which is why the role charts show a
  large "Other" group.

## Method

| Step | What it does |
|---|---|
| `clean` | Collapse whitespace, drop duplicate ids and descriptions under 50 characters. |
| Skills | Word-bounded regex taxonomy (`src/jobinsights/taxonomy.py`) of ~100 skills in 11 families. Ambiguous names (R, Excel, SAS, SAP, Go) are case-sensitive and need context. The original notebook's substring matcher is kept as `legacy_skills` so the evaluation can measure it. |
| Salary | Parsed from the text (ranges, `$85-120K`, hourly/weekly/monthly) and from the structured pay columns, with mislabelled periods repaired, then annualised to USD. |
| Work authorisation | Rule set (`visa.py`) with four classes: friendly (sponsorship offered or OPT / H-1B / any visa welcomed), restricted (no sponsorship, citizen / green-card / clearance only), mixed, not stated. Also flags OPT / F-1 mentions and contract / staffing-style postings (W2, C2C, 1099, "our client"). A mention of OPT can be a restriction, so the dashboard filter is a prompt to read the posting. |
| Work arrangement | Rule set with four classes (remote, hybrid, onsite, unknown) that handles negation ("not a remote role") and non-location uses ("remote sensing", "remote teams"). |
| LLM extraction | A free open-weights instruct model (`Qwen/Qwen2.5-1.5B-Instruct`) run locally on CPU with a JSON prompt. No paid API is used. Outputs that do not parse are counted. |
| Evaluation | 150 postings drawn with a fixed seed in three strata (tech titles, remote keywords, random), labelled by hand for work arrangement, stated base pay and required skills. Field-level precision / recall / F1 with 95% bootstrap intervals over postings. |
| Clusters | Hard skills only (soft skills appear in nearly every posting), IDF-weighted, SVD to 15 dimensions, KMeans. k is chosen from a sweep using silhouette among partitions that are stable across seeds (ARI). Clusters are named by their most distinctive skills. |
| Salary model | Ridge and gradient boosting on title TF-IDF, skills, role family, experience, state, work type; compared with median baselines; out-of-fold errors with folds grouped by normalised title so repeated postings cannot leak. |

## Limitations

* **Hand-labelled evaluation is small.** 150 postings gives wide confidence intervals, and the sample is
  stratified toward tech titles and remote keywords, so scores describe those strata, not the whole
  dataset. One person labelled the data, so there is no inter-annotator agreement estimate.
* **The LLM comparison is for one small local model** (1.5B parameters, CPU, greedy decoding). It says
  nothing about larger hosted models; no paid API was used or evaluated.
* **Rules were written while inspecting postings from this same dataset.** The 150 evaluation postings were
  drawn at random and not picked for the rules, but I did not hold them out from my inspection, so the
  scores are not a clean held-out estimate and may not carry over to other sources of postings.
* **Salary:** only postings that state pay are modelled, and they are not a random sample; the model
  describes pay-transparent postings. Pay is not imputed for the others. Hourly pay is annualised at
  2,080 hours, so part-time roles are overstated. The structured pay columns contain errors that the
  normaliser repairs heuristically (an annual figure tagged hourly, etc.).
* **Remote status:** the text often does not say ("unknown" is the largest class). The platform's own
  `remote_allowed` flag is present on only a minority of rows and is not used as a label.
* **Clusters** are descriptive. Silhouette is moderate, stability is below 0.8 ARI at the chosen k, and
  the original method's silhouette is measured in a different (raw) space, so the two silhouettes are not
  strictly comparable; the ARI stability and the interpretable cluster names are the more useful signal.
* **Dataset size:** numbers in the results block come from the 33k-row public subset unless the block says
  otherwise. Re-running `jobinsights run-all` on the full 123,849 rows will change them.

* **Work authorisation:** only about 5% of postings say anything about it, so an "OPT-friendly" filter
  returns few postings. It is evaluated by precision on flagged postings only; recall is not estimable. It is
  a text signal, not a list of H-1B sponsors: company-level sponsorship history (for example the public USCIS
  H-1B Employer Data Hub or DOL LCA disclosure data) is not joined yet, and the public 33k subset has no
  company names, only ids. The full Kaggle file includes `company_name`, which would allow that join.

## Project layout

```
src/jobinsights/   data, taxonomy, remote, salary, extract, llm, evalset, evaluate,
                   cluster, salary_model, roles, report, label_app, dashboard_app, cli
tests/             pytest for parsers, normalisers, metrics, clustering helpers
docs/              UPGRADE_PLAN.md (audit + plan), dashboard.png
outputs/           RESULTS.md and JSON results written by the pipeline
notebooks/legacy/  the original prototype notebooks and plots (not reproducible, kept for history)
```

## License

MIT License
