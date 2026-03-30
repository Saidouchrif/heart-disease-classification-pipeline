# Heart Disease Classification Pipeline

Projet de machine learning de bout en bout pour predire la presence d'une maladie cardiaque a partir de variables cliniques.

## 1. Objectif du projet

Ce projet couvre tout le cycle d'un projet data science:

- comprehension des donnees
- analyse exploratoire (EDA)
- nettoyage et preparation
- feature engineering
- entrainement du modele
- evaluation
- mise en production simple avec Streamlit

Le modele final sauvegarde est une **LogisticRegression** (`models/heart_model.pkl`) avec un **StandardScaler** (`models/scaler.pkl`).

## 2. Structure du projet

```text
heart-disease-classification-pipeline/
|-- Data/
|   |-- Heart_Disease_Prediction.csv
|-- Analayse/
|   |-- heart_disease_pipeline.ipynb
|-- models/
|   |-- heart_model.pkl
|   `-- scaler.pkl
|-- templates/
|   `-- app.py
|-- Visualisation/
|   `-- heart_dataset_explanation.docx
|-- requirements.txt
`-- README.md
```

## 3. Variables utilisees

Variables du dataset:

- `Age`
- `Sex`
- `Chest pain type`
- `BP`
- `Cholesterol`
- `FBS over 120`
- `EKG results`
- `Max HR`
- `Exercise angina`
- `ST depression`
- `Slope of ST`
- `Number of vessels fluro`
- `Thallium`
- `Heart Disease` (target)

Feature cree pendant la preparation:

- `age_group = Age // 10`

Important: le scaler a ete entraine avec **14 features** (les 13 variables d'entree + `age_group`).

## 4. Installation

Depuis la racine du projet:

```bash
pip install -r requirements.txt
```

## 5. Lancer l'application Streamlit

Commande recommandee (depuis la racine du projet):

```bash
streamlit run templates/app.py
```

L'application ouvre une interface pour saisir les variables patient et obtenir:

- la prediction (presence/absence)
- la probabilite associee

## 6. Erreur resolue: `X has 13 features, but StandardScaler is expecting 14`

### Cause

`app.py` envoyait 13 valeurs au scaler alors que le scaler attend 14 colonnes (dont `age_group`).

### Correction appliquee

- ajout de `age_group` calcule dans l'app (`age // 10`)
- creation de l'entree sous forme de **DataFrame avec noms de colonnes**
- alignement de l'ordre des colonnes avec `scaler.feature_names_in_`

Cette correction elimine:

- l'erreur de dimension (`13 vs 14`)
- l'avertissement `X does not have valid feature names`

## 7. Workflow recommande

1. Explorer et preparer les donnees dans le notebook (`Analayse/heart_disease_pipeline.ipynb`).
2. Entrainement + sauvegarde du modele et scaler dans `models/`.
3. Tester localement l'app Streamlit.
4. Iterer sur les features/modeles selon les performances.

## 8. Limites et bonnes pratiques

- Ce projet est pedagogique et ne remplace pas un diagnostic medical.
- Toujours valider le modele sur des donnees non vues.
- Versionner les artefacts (`model`, `scaler`) en meme temps que le code d'inference.

## 9. Auteurs / Credits

Projet realise pour la pratique d'un pipeline ML de classification en sante.
