# Introduction
### This repo contains files for the hierarchical classifier:
### *./amazon* - folder with the dataset
### *hierarchical_classifier.py* - source code of the hierarchical classifier.
### *classifier_fast_api.py* - FastAPI for using the trained hierarchical classifier
### *main.py* - program in which I performed profiling and small tests.
### *profile_output.txt* - results of profiling using line_profiler
### *hierarchical_classifier_notebook.ipynb* - Jupyter notebook, which contains various experiments with the hierarchical and flat classifier with a detailed text description.
### *Dockerfile* - Docker file for running the FastAPI application
### *requirements.txt* - required dependencies for building the Docker image and running the Docker container.

# Hierarchical classifier
### I wrote a simple hierarchical classifier that models a distribution of the form:
```
P(Cat1, Cat2, Cat3) = P(Cat3 | Cat1, Cat2)*P(Cat2 | Cat1)*P(Cat1)
```
### A more detailed description of how this classifier works can be found in the corresponding file *hierarchical_classifier.py*.
### Experiments with the hierarchical classifier and its comparison with the flat classifier can be found in the jupyter notebook *hierarchical_classifier_notebook.ipynb*.

# FastAPI
The file *classifier_fast_api.py* contains a small FastAPI for interacting with the trained hierarchical classifier - it takes a post request with text in JSON as input, and the output is categories of the following type:
```
"Cat1": cat_1_model_prediction,
"Cat2": cat_2_model_prediction,
"Cat3": cat_3_model_prediction
```
For a separate check, I ran it on a local host using:
```bash
uvicorn classifier_fast_api:app --host 0.0.0.0 --port 8000
```
And then in the desktop version of Postman I sent a request to the address:
```
http://127.0.0.1:8000/predict/
```
Feeding JSON of the following type as input:
```json
{
"text": "The description and photo on this product needs to be changed to indicate this product is the BuffalOs version of this beef jerky."
}
```

# Dockerfile
### The ***Dockerfile*** contains the installation of all the necessary dependencies specified in ***requirements.txt***.
### While in the directory with all the necessary files, I built the docker image simply with:
```bash
docker build -t my-fastapi-app .
```
### And then launched the container on the local host with:
```bash
docker run -d -p 8000:8000 my-fastapi-app
```
### After that, requests can be sent in the same way using Postman according to the scheme indicated above.