# Course alignment — DataTalksClub ML Zoomcamp midterm

Verified from the official `projects/` rubric in the course repo (same rubric for midterm and
both capstones). **This is the real rubric, not my invention.** I could *not* confirm the
2026 cohort's submission **deadline** from a machine-readable page (it lives in the course
management system / a Google Sheet), so **no deadline is assumed here** — check
`courses.datatalks.club/ml-zoomcamp-2026/` yourself.

## Evaluation criteria (max 16 points)

| Criterion | Pts | What earns full marks | How this project targets it |
|---|---:|---|---|
| Problem description | 2 | README gives enough context to show the problem and how the solution is used | README + DATA.md |
| EDA | 2 | Extensive: ranges, missing values, **target analysis**, feature importance | inspection + notebook (Milestone 2+) |
| Model training | 3 | **Multiple models incl. linear and tree-based, with tuned parameters** | logistic reg + decision tree + random forest, validation-tuned |
| Exporting notebook to script | 1 | Training logic in a separate script | `src/.../train.py` |
| Reproducibility | 1 | Notebook/script run without errors; data committed **or** clear download instructions | `scripts/get_data.py` + fixed seeds |
| Model deployment | 1 | Served with Flask / BentoML / similar web framework | **FastAPI** service |
| Dependency & env management | 2 | Dependency file **+** virtual env, README explains install/activate | `requirements.txt` + venv instructions |
| Containerization | 2 | Dockerfile, README explains build & run | Dockerfile (Milestone later) |
| Cloud deployment | 2 | Deploy docs/code **+** a test URL or video/screenshot | optional stretch goal |

**Peer review:** you must review **3** peers' projects or you fail the project; each review = +3 pts.

## Deliverables the rubric expects
- README with problem description + run instructions.
- Dataset committed **or** clear download instructions. (We use download instructions — data is git-ignored.)
- Notebook: data prep, EDA, feature importance, model selection + tuning.
- `train.py`: trains the final model and saves it.
- `predict.py` (or FastAPI app): loads the model and serves predictions.
- Dependency file(s): `requirements.txt` (or Pipenv / conda / pyproject).
- Dockerfile for the service.
- Deployment: a URL, or a video/image of the deployed service in use.

## Rules worth noting
- **Disallowed datasets:** car price, customer/Telco churn, credit risk scoring, clothing classification.
  (Online Retail repeat-purchase is **not** on that list — allowed.)
- Any library is fine if documented for reviewers.
- Plagiarism (incl. reusing your own prior projects) scores 0.
- `nbviewer.org` is recommended if GitHub fails to render a notebook.
