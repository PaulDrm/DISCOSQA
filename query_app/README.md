# ESA Intelligent and Operational Assistant

The aim of this repository is to collect the source code of providing an assistant 
to answer natural language queries by accessing multiple database sources. 

## How to currently use this repository

#### Run locally 
*You have to have a local python environment installed (V3.8, and V3.7 tested)*

- Clone repository locally 
- Navigate with commandline tool to `./query_app/app`
- Install dependencies with `pip install -r ./requirements.txt`
- download files from https://strathcloud.sharefile.eu/d-s14533191c46f4b768c588d74556048dc
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG`
- run Streamlit app with `streamlit run app.py`
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

#### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
- download files from https://strathcloud.sharefile.eu/d-s14533191c46f4b768c588d74556048dc
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG`
- Navigate in commandline tool to `./query_app`
- run `docker build . -t streamlit_app`
- run `docker run -p 8501:8501 streamlit_app`
- Open an internet browser and type `localhost:8501` in address line to get to app

