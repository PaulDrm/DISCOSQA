# ESA Intelligent and Operational Assistant

The aim of this repository is to collect the source code of providing an assistant 
to answer natural language queries by accesing multiple database sources. 

## How to currently use this repository

- Clone locally repository
- Install dependencies (Pytorch, Huggingface Transformers, Streamlit, etc)
- download files from https://strathcloud.sharefile.eu/d-s14533191c46f4b768c588d74556048dc
- Insert `entity_embeddings.pt` in `query_app\processed\entity` and `esa_kb.json` in `query_app\Preprocessing_KG`
- run Streamlit app with `streamlit run app.py`