# Hosted demo

`streamlit_app.py` is a self-contained explorer for the findings. It reads only the aggregate files in
`data/` (counts, shares and medians per group; no postings, no description text, no links) and needs
just `streamlit`, `plotly`, `pandas` and `numpy` (`requirements.txt`).

Rebuild the data bundle from a full pipeline run:

```bash
jobinsights run-all
jobinsights export-demo      # writes demo/data/ (about 1 MB)
streamlit run demo/streamlit_app.py
```

Deploy on Streamlit Community Cloud (free): sign in at share.streamlit.io with GitHub, choose
**Create app**, pick this repository, branch `main`, main file `demo/streamlit_app.py`. Cloud installs
`demo/requirements.txt`; no secrets or data downloads are needed.
