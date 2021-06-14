# COP-E-CAT
In order to ensure that analyses of complex electronic healthcare record (EHR) data are reproducible and generalizable, it is crucial for researchers to use comparable preprocessing, filtering, and imputation strategies. We introduce COP-E-CAT: Cleaning and Organization Pipeline for EHR Computational and Analytic Tasks, an open-source organization and analysis software for MIMIC-IV, a ubiquitous benchmark EHR dataset. COP-E-CAT allows users to select filtering characteristics and preprocess covariates to generate data structures for use in downstream analysis tasks. This user-friendly approach shows promise in facilitating reproducibility and comparability among studies that leverage MIMIC-IV, and enhances EHR accessibility to a wider spectrum of researchers than current data processing methods. We demonstrate the versatility of our workflow by describing three use cases: ensemble learning for decompensation prediction, reinforcement learning for clinical decision support, and dimension reduction.
The paper is available at https://doi.org/10.1145/3459930.3469536.

## Workflow
- Get access to the MIMIC-IV dataset and configure a SQL server to query the dataset.
- Modify the SQL configuration in cop_e_cat/utils/pipeline_config.py
- Instantiatiate a Cop-E-Cat class and generate state spaces. Examples are in vignettes/

