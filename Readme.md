# **Credit Risk – Projet**

## 1. Introduction

L’évaluation du risque de crédit est centrale en FinTech : elle permet aux banques et plateformes de prêt de savoir si un client a de fortes chances de rembourser.
Aujourd’hui, l’IA joue un rôle clé, parce qu’elle apprend à partir des données (revenu, historique, comportement…) pour améliorer la qualité des décisions financières.
Elle permet d’analyser rapidement de grandes quantités de données et d’améliorer la précision des décisions de prêt, ce qui aide à mieux identifier les clients à risque.

---

## 2. Objectifs de cette analyse

* Identifier les facteurs qui influencent le risque de défaut.
* Préparer et nettoyer un dataset de risque de crédit.
* Construire deux modèles de classification (Logistic Regression & Random Forest).
* Évaluer les performances des modèles et interpréter les erreurs.
* Comparer les modèles et choisir celui qui prédit le mieux le risque.

---

## 3. Dataset Overview

**Dataset :** `credit_risk_dataset.csv`

**Colonnes principales :**

* `person_age` : âge
* `person_income` : revenu
* `person_home_ownership` : type de logement
* `person_emp_length` : ancienneté professionnelle
* `loan_intent` : but du prêt
* `loan_grade` : grade de crédit
* `loan_amnt` : montant du prêt
* `loan_int_rate` : taux d’intérêt
* `loan_status` : cible (0 = remboursé, 1 = défaut)
* `loan_percent_income` : ratio prêt/revenu
* `cb_person_default_on_file` : défaut passé
* `cb_person_cred_hist_length` : longueur historique crédit

---

## 4. Column Data Types

* **Numériques :** age, income, emp_length, loan_amnt, loan_int_rate…
* **Catégorielles :** loan_intent, home_ownership, grade, default_on_file…

---

## 5. Importation des bibliothèques

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
```

---

## 5.1 Préparation des données

Chargement du dataset, exploration, nettoyage.

---

## 6. Exploration des données

### 6.1 Analyse descriptive

Statistiques, valeurs manquantes et aperçu du dataset.

### 6.2 Visualisation de la variable cible

Graphique : distribution du `loan_status`.

**Interprétation :**

* 80% : prêts remboursés
* 20% : prêts en défaut
* Dataset très déséquilibré → nécessite techniques adaptées (class_weight, SMOTE…)

---

## 7. Visualisation des relations

Exemple : boxplot âge vs statut de risque.

**Analyse :**

* L’âge ne semble pas discriminant
* Présence de valeurs aberrantes (âges > 80 → erreurs de saisie)

---

## 8. Prétraitement des données

### 8.1 Valeurs manquantes

* Médiane pour numériques
* Mode pour catégorielles

### 8.2 Encodage des variables catégorielles

One-hot encoding via :

```python
pd.get_dummies(..., drop_first=True)
```

### 8.3 Standardisation

Normalisation via `StandardScaler`.

### 8.4 Division du dataset

Split 80% / 20% avec stratification.

---

## 9. Modélisation IA

### 9.1 Régression Logistique

```python
lr_model = LogisticRegression(class_weight='balanced')
```

**Performances :**

* Accuracy ≈ **0.81**
* Bonne précision sur classe 0
* Recall élevé sur classe 1 grâce au balancing

---

### 9.2 Random Forest

```python
rf_model = RandomForestClassifier(class_weight='balanced')
```

**Performances :**

* Accuracy ≈ **0.93**
* Très bon recall sur classe 0
* Meilleur modèle globalement

---

### 9.3 Matrice de confusion

RF performe très bien mais encore :

* **Faux négatifs → 405**
  → Risque réel pour une banque (clients risqués prédits comme sûrs)

---

## 10. Importance des variables

Top 5 features (Random Forest) :

1. `loan_percent_income`
2. `person_income`
3. `loan_int_rate`
4. `loan_amnt`
5. `loan_grade_D`

---

## 11. Interprétation FinTech

Dans le scoring crédit :

* **Faux positifs = les plus dangereux**

  > Accorder un prêt à un client qui fera défaut → perte directe.
* **Faux négatifs**

  > Manque à gagner seulement.

Le modèle RF minimise bien les faux positifs → bon modèle pour une banque.
