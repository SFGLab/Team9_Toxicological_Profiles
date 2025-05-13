# Team9_Toxicological_Profiles

## Important links
* link to the [dataset](https://drive.google.com/drive/folders/195KAyBS80Qdu5-uTHUWGVScDd4S7jBmM)
* [introduction to the problem](https://docs.google.com/presentation/d/1WYebbOqxnCUWdD_irGYNAFhBpz03HkJezKHb4039Ud0/edit#slide=id.g357624754e3_0_69)
* Simmilar solutions: [https://paperswithcode.com/sota/drug-discovery-on-tox21](https://paperswithcode.com/sota/drug-discovery-on-tox21)

## Alternative Solutions
Based on the [Tox21 Dataset](https://tripod.nih.gov/pubdata/) there is [Therapeutics Data Commons (TDC)](https://arxiv.org/pdf/2102.09548v2) dataset, and [Tox24 Chalange](https://ochem.eu/static/challenge-data.do) dataset.<br>
Tox21:
* [XGBoost](https://arxiv.org/pdf/2204.07532v3):

Tox24:
* [Keggle competitors](https://www.kaggle.com/datasets/antoninadolgorukova/tox24-challenge-data/code)
* [XGBoost](https://arxiv.org/pdf/2204.07532v3):
  * Features:
    1. Fingerprints: ErG, Mordred, Pubchem, MACCS, RDKit
    2. Fingerprints: 'fgr', 'datamol', 'ALogPSOEState', 'Mold2', 'SIRMSmix', 'MAP4', 'atombond', 'estate', 'JPlogP', 'ISIDAfragments'
  * algorithms:
    1. Cross-validation using XGBoost
    2. XGBoost

[Best TDC models according to Papers With Code](https://paperswithcode.com/dataset/tdcommons):
* [MapLight](https://arxiv.org/pdf/2310.00174v1)
  * Features:
    1. fingerprints: extended-connectivity fingerprints (ECFP), Avalon, and extended reduced graph approach (ErG),
    2. 200 molecular properties, including: "number of rings, molecular weight"
  * algorithms:
    1. Parameter search via grid search CV,
    2. CatBoost,
    3. LightGBM
* [XGBoost](https://arxiv.org/pdf/2204.07532v3):
  * Features:
    1. Fingerprints: MACCS, ECFP, Mol2Vec,
    2. Descriptors: Pubchem (881 distinct structural features), Molecular Access System (Mordred), RDKit
  * algorithms:
    1. Parameter search via randomized grid search CV,
    2. Extreme Gradient Boosting (XGBoost)
