# ESA's DISCOS QA system

This is the source for the ACL paper submission "DISCOSQA: A Knowledge Base Question Answering System for Space Debris based on Program Induction".

# TRAINING
Files are already processed in training and validation split. 

Navigate into ./app folder

First train on original KQA Pro dataset

Note: --input_dir has to be adapted to model type (rob = RoBERTa) 

python -m Pretraining.train_rob --input_dir ./kqa_processed_ent_rob --output_dir ./train_ent_cosmicrob --save_dir ./train_ent_cosmicrob --model_name_or_path icelab/cosmicroberta --val_batch_size=256 --train_batch_size=128 --learning_rate=1e-5 --save_steps=2400 --logging_steps=1200 --num_train_epochs=100 --eval_steps=1200 --wandb=0 --model_type roberta

### Pick a good checkpoint and fine-tune on DWQ dataset

python -m Pretraining.train --input_dir ./processed_dwq_rob --output_dir ./cos_dwq --save_dir ./cos_dwq --model_name_or_path enter-check-point --val_batch_size=256 --train_batch_size=128 --learning_rate=1e-5 --save_steps=120 --logging_steps=2 --eval_steps=60 --num_train_epochs=80 --wandb=0 --model_type roberta --crf_learning_rate 1e-4
  

For evaluation run: 

python -m Pretraining.eval --models_dir trained models directory --data_dir ./processed_dwq_rob

# APP


The aim of this repository is to collect the source code of providing an assistant 
to answer natural language queries on the DISCOS database. 

## How to currently use this repository

#### Run locally 
*You have to have a local python environment installed (V3.8, and V3.7 tested)*

- Clone repository locally 
- Navigate with commandline tool to `./query_app/app`
- Install dependencies with `pip install -r ./requirements.txt`
  
    in respective folders  
- Steps for using SUTIME (getting time values from query): Download maven from https://maven.apache.org/download.cgi
- Follow installation steps on https://maven.apache.org/install.html (Set path variables accordingly )
- run Streamlit app with `streamlit run app.py`
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

#### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
    in respective folders 

- Navigate in commandline tool to `./query_app`
- run `docker build . -t ioa_demo_app`
- run `docker run -p 8501:8501 --name ioa_streamlit_demo ioa_demo_app`
- Open an internet browser and type `localhost:8501` in address line to get to app



