# ESA Intelligent and Operational Assistant

The aim of this repository is to collect the source code of providing an assistant 
to answer natural language queries by accessing multiple database sources. 

## How to currently use this repository

#### Run locally 
*You have to have a local python environment installed (V3.8, and V3.7 tested)*

- Clone repository locally 
- Navigate with commandline tool to `./query_app/app`
- Install dependencies with `pip install -r ./requirements.txt`
- download files from https://strathcloud.sharefile.eu/d-s8a6ffa8c87b444ee94f6de7533f920a5
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG` manually or run
-  1.  curl 'https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dte59a95ea62094ce28b527fd70e35218d&cid=p-IlUNJ8izgjuaDrQX0DYg&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666888863&zsid=185A&h=MxfIl9ppitFmvZVxY+CexDs1vm0r3Y5hrUbMPmhGjIo=' -O entity_embeddings.pt
   2. curl 'https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dtdbfe28aa69d14ce9920c5a6a1a37f6ff&cid=9wnqzdZ0WU8efTBtEKB5pQ&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1667475475&zsid=185A&h=mOLn0nmH2cdfEOYUwy4q36tYJo%2FMh4Jp6fZp5aLxDxQ%3D' -O esa_kb.json

    in respective folders  
- run Streamlit app with `streamlit run app.py`
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

#### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
- download files from https://strathcloud.sharefile.eu/d-s8a6ffa8c87b444ee94f6de7533f920a5
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG`
-   1.  curl 'https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dte59a95ea62094ce28b527fd70e35218d&cid=p-IlUNJ8izgjuaDrQX0DYg&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1666888863&zsid=185A&h=MxfIl9ppitFmvZVxY+CexDs1vm0r3Y5hrUbMPmhGjIo=' -O entity_embeddings.pt
   2. curl 'https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dtdbfe28aa69d14ce9920c5a6a1a37f6ff&cid=9wnqzdZ0WU8efTBtEKB5pQ&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1667475475&zsid=185A&h=mOLn0nmH2cdfEOYUwy4q36tYJo%2FMh4Jp6fZp5aLxDxQ%3D' -O esa_kb.json
  
    in respective folders 

- Navigate in commandline tool to `./query_app`
- run `docker build . -t ioa_demo_app`
- run `docker run -p 8501:8501 --name ioa_streamlit_demo ioa_demo_app`
- Open an internet browser and type `localhost:8501` in address line to get to app

#### Evaluation

- Clone repository locally
- download files at https://strathcloud.sharefile.eu/d-s8a6ffa8c87b444ee94f6de7533f920a5
- insert `entity_embeddings_3110.pt` in `query_app/app/test_data/entity`
-  or download with `curl 'https://its-szc-b.ds.strath.ac.uk/download.ashx?dt=dt4fa873e2b9ad41c5a3d62278907fda43&cid=iSYyA09tAQTO8nC87QylNQ&zoneid=z1ed8181-1d10-40e2-a101-4388ad15b0e5&exp=1667476782&zsid=185A&h=Iz0bFf%2B8vm5aE2jSbhPaqyqdAlTDAVJU%2Bp%2FkIVt%2F%2BEs%3D' -o entity_embeddings_3110.pt`
  in respective folder
- from ./app folder run `python -m Pretraining.train_eval --input_dir ./test_data --output_dir ./evaluate --save_dir ./evaluate --model_name_or_path PaulD/IOA_261022-11999 --val_batch_size=8`

