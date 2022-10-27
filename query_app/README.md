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
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG` manually or run
-  1.  wget "https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dte59a95ea62094ce28b527fd70e35218d&cid=p-IlUNJ8izgjuaDrQX0DYg&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666888863&zsid=185A&h=MxfIl9ppitFmvZVxY+CexDs1vm0r3Y5hrUbMPmhGjIo=" -O entity_embeddings.pt
   2. wget "https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dt3580695fd4274f83b574dae5de057913&cid=EjfsbKd-zuWfSFsB6K6jeA&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666890615&zsid=185A&h=UetjBwn7zvSQeOZzvgsSqfK609z%2BNYC06gNcQdloScI%3D" -O esa_kb.json
  
    in respective folders  
- run Streamlit app with `streamlit run app.py`
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

#### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
- download files from https://strathcloud.sharefile.eu/d-s14533191c46f4b768c588d74556048dc
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG`
-   1.  wget "https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dte59a95ea62094ce28b527fd70e35218d&cid=p-IlUNJ8izgjuaDrQX0DYg&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666888863&zsid=185A&h=MxfIl9ppitFmvZVxY+CexDs1vm0r3Y5hrUbMPmhGjIo=" -O entity_embeddings.pt
   2. wget "https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dt3580695fd4274f83b574dae5de057913&cid=EjfsbKd-zuWfSFsB6K6jeA&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666890615&zsid=185A&h=UetjBwn7zvSQeOZzvgsSqfK609z%2BNYC06gNcQdloScI%3D" -O esa_kb.json
  
    in respective folders 

- Navigate in commandline tool to `./query_app`
- run `docker build . -t ioa_demo_app`
- run `docker run -p 8501:8501 --name ioa_streamlit_demo ioa_demo_app`
- Open an internet browser and type `localhost:8501` in address line to get to app

