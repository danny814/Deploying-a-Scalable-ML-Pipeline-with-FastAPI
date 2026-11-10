# Deploying a Scalable ML Pipeline with FastAPI

[![Python CI](https://github.com/danny814/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/manual.yml/badge.svg?branch=main)](https://github.com/danny814/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/manual.yml?query=branch%3Amain)

## Continuous Integration

- **Repository:** https://github.com/danny814/Deploying-a-Scalable-ML-Pipeline-with-FastAPI
- **Workflow:** `.github/workflows/manual.yml` runs flake8 and pytest on every push to `main`/`master`.
- **Passing Actions run:** https://github.com/danny814/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/runs/42774264700
- **Commit verified by that run:** f4c4287e4b8ccece62669c680f0c50b803ff51fa
- **All runs on `main` (latest at the top):** https://github.com/danny814/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/actions/workflows/manual.yml?query=branch%3Amain
- Screenshot of a passing build: `screenshots/continuous_integration.png`

---

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
