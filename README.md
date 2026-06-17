# Movie Rating Prediction - ML Design Project

This project explores a TMDB-style movie dataset and builds machine learning models to predict whether a movie is highly rated. The main notebook is `movie_ml_project.ipynb`; `run_movie_project.py` is a reproducible script version that regenerates the saved tables and figures.

## Project Goal

The core question is:

> Can movie metadata such as popularity, vote count, release year, genre, and original language predict whether a movie has a high audience rating?

The project frames this as a binary classification task:

- `High_Rated = 1` when `Vote_Average >= 7.0`
- `High_Rated = 0` otherwise

It also includes a small regression baseline for predicting the continuous `Vote_Average`.

## Dataset

The dataset is the included local file `mymoviedb.csv`, a TMDB-style movie metadata table with about 9,800 movies. It includes:

- `Release_Date`
- `Title`
- `Overview`
- `Popularity`
- `Vote_Count`
- `Vote_Average`
- `Original_Language`
- `Genre`
- `Poster_Url`

The original external download/source link is not stored in this repository, so the project documents the source conservatively as the included local TMDB-style CSV file.

## What The Project Covers

- Data loading and cleaning
- Exploratory visualisation of ratings, popularity, release year, and genres
- Explicit assumptions and preconceptions before modeling
- Feature engineering with log-transformed popularity/vote counts, genre indicators, and language indicators
- PCA dimensionality reduction
- k-means clustering and cluster profiling
- Classification model comparison:
  - Logistic Regression with L2 regularisation
  - SVM with RBF kernel
  - Random Forest
  - CatBoost, or HistGradientBoosting if CatBoost is unavailable
  - MLP neural network
- Model selection with 5-fold cross-validation and a held-out test split
- ROC curve, confusion matrix, and model comparison plots
- Feature importance analysis
- Ridge regression baseline for continuous rating prediction

## Key Files

- `movie_ml_project.ipynb` - main notebook submission
- `run_movie_project.py` - script version of the workflow
- `mymoviedb.csv` - movie dataset
- `classification_results.csv` - classifier comparison results
- `kmeans_metrics.csv` - k-means inertia and silhouette scores
- `kmeans_cluster_profiles.csv` - cluster summaries
- `figures/` - generated plots

## How To Run

Create and activate a Python environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the script:

```powershell
python run_movie_project.py
```

Or open and run the notebook:

```powershell
jupyter notebook movie_ml_project.ipynb
```

On this machine, the Python launcher `py -3` is not configured in the sandbox, but the project was verified with:

```powershell
& "C:\Users\keren segev\AppData\Local\Programs\Python\Python311\python.exe" run_movie_project.py
```

If CatBoost is unavailable, the code automatically falls back to scikit-learn's `HistGradientBoostingClassifier`.

## Current Result Summary

The saved classifier results show that CatBoost is the strongest model overall in this run, with the best cross-validated ROC-AUC and held-out test ROC-AUC. Random Forest and MLP are competitive, while Logistic Regression provides a useful simpler baseline.

The clustering and PCA views suggest that movie groups overlap heavily: genre and popularity patterns matter, but high-rated movies are not cleanly separated by a single unsupervised cluster. This supports using supervised nonlinear models for the main prediction task.
