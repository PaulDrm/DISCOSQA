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
-  1. `curl 'https://groundflex.sharepoint.com/sites/ESAIOA/_layouts/15/download.aspx?UniqueId=8517f6ec%2D785b%2D4bcd%2Dbaf9%2D3fb8b392a729' -o entity_embeddings.pt`
   2. `curl 'https://groundflex.sharepoint.com/sites/ESAIOA/_layouts/15/download.aspx?UniqueId=1640221151a7487f8aba59a42f49ce5b&e=mW99HU' -O esa_kb.json`
  
    in respective folders 
    in respective folders  
- run Streamlit app with `streamlit run app.py`
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

#### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
- download files from https://strathcloud.sharefile.eu/d-s8a6ffa8c87b444ee94f6de7533f920a5
- Insert `entity_embeddings.pt` in `query_app/app/processed/entity` and `esa_kb.json` in `query_app/app/Preprocessing_KG` manually *or*
-   1. `curl 'https://groundflex.sharepoint.com/sites/ESAIOA/_layouts/15/download.aspx?UniqueId=8517f6ec%2D785b%2D4bcd%2Dbaf9%2D3fb8b392a729' -o entity_embeddings.pt`
   2. `curl 'https://groundflex.sharepoint.com/sites/ESAIOA/_layouts/15/download.aspx?UniqueId=1640221151a7487f8aba59a42f49ce5b&e=mW99HU' -O esa_kb.json`
  
    in respective folders 

- Navigate in commandline tool to `./query_app`
- run `docker build . -t ioa_demo_app`
- run `docker run -p 8501:8501 --name ioa_streamlit_demo ioa_demo_app`
- Open an internet browser and type `localhost:8501` in address line to get to app

#### Evaluation

- Clone repository locally
- download files at https://strathcloud.sharefile.eu/d-s8a6ffa8c87b444ee94f6de7533f920a5
- insert `entity_embeddings_3110.pt` in `query_app/app/test_data/entity`
-  or download with `curl 'https://groundflex.sharepoint.com/sites/ESAIOA/_layouts/15/download.aspx?UniqueId=8517f6ec%2D785b%2D4bcd%2Dbaf9%2D3fb8b392a729' -o entity_embeddings_3110.pt`

  in respective folder
- from ./app folder run `python -m Pretraining.train_eval --input_dir ./test_data --output_dir ./evaluate --save_dir ./evaluate --model_name_or_path PaulD/IOA_261022-11999 --val_batch_size=8`

