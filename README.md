# Bourse

Projet minimal de prédiction des actions boursières en Python.

## Objectif

Fournir une base simple pour :
- générer des données de prix synthétiques,
- créer des variables explicatives (features),
- entraîner un modèle de régression linéaire,
- produire une prédiction de prix du lendemain.

## Lancer une prédiction

```bash
python -m bourse.predict
```

Exemple avec paramètres personnalisés :

```bash
python -m bourse.predict --days 180 --window 7 --seed 123
```

## Exécuter les tests

```bash
python -m unittest discover -s tests -v
```