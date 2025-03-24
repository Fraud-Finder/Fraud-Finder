# Fraud-Finder
End-to-end MLOps project using GCP for credit card fraud detection.


#### Folder Descriptions

- `data` – Dataset location (linked from GCS; not tracked in Git)
- `scripts` – Python scripts for preprocessing, training, and prediction
- `models` – Local model copies (for testing or demo without Cloud Run)
- `infra` – Dockerfile, CI/CD configs, Terraform
- `notebooks` – Prototyping or experimentation using Jupyter
- `tests` – Unit and integration tests for pipeline components
- `dashboard` – Looker Studio files or screenshots
- `docs` – Changelogs, team notes, or planning


## Cloning the Repo and Virtual Environment

```bash
##bash
git clone https://github.com/Stinly/Fraud-Finder.git
cd Fraud-Finder


##### Virtual Enviroment for dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

