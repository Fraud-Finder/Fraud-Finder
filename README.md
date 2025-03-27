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
```
## Accessing BigQuery Data (For Teammates)

To access the project datasets:

1. Make sure you've been added to the GCP project: `fraud-finder-454706`
2. Open BigQuery in Google Cloud Console
3. Navigate to dataset `fraud_data`
4. Query tables: `train_data`, `validation_data`, `test_data`

If you get a permissions error, message Austin to be added as a BigQuery Data Viewer.

## BigQuery Tables

The following datasets have been created from `transactions_cleaned` using deterministic fingerprint-based splitting:

- `fraud_data.train_data` — 70% of the data, used for training
- `fraud_data.validation_data` — 10%, used for hyperparameter tuning
- `fraud_data.test_data` — 20%, used for final evaluation

All datasets are stored in BigQuery under the GCP project: `fraud-finder-454706`.


