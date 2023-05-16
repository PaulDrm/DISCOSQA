# ESA's DISCOS QA system

This is the source code for the ACL industry track paper submission "DISCOSQA: A Knowledge Base Question Answering System for Space Debris based on Program Induction".

# INSTALLATION

Clone the repository with Git Large File Storage (LFS) enabled by running first `git clone https://github.com/PaulDrm/DISCOSQA.git`, then navigating into the project's folder and subsequently pulling large files with `git lfs pull`

Also create a virtual python environment and install the required dependencies with `pip install -r app/requirements.txt` 

# PREPROCESSING

The main script for preprocessing is preprocess.py. As prerequeisite, it takes that the training dataset has already been split into training and validation. 
Two modes exists for preprocessing. One for encoding the database entries, the training as well as the validation dataset for training ("-- mode train"). The second mode of preprocessing is for encoding the database in case it gets updated or a if a new version of the model. Moreover, is will create a file called `entity_embeddings.pt` in `<output_dir>/entity`, for that model and version of the database which is important for fast inference as the model can load in the precomputed embeddings into memory during inference and does not have compute them again. 


Example command for running preprocessing script for DWQ dataset and training:
`python -m Pretraining.preprocess --input_dir ./DWQ --output_dir ./dwq_processed_rob --train_file_path ./DWQ/train.json --valid_file_path ./DWQ/valid.json --model_type roberta --mode train --kb esa_kb.json --model_name <enter_model_name>`

Example command for running preprocessing script for DWQ dataset and inference:
`python -m Pretraining.preprocess --input_dir ./DWQ --output_dir ./dwq_processed_rob --train_file_path ./DWQ/train.json --valid_file_path ./DWQ/valid.json --model_type roberta --mode inference --kb esa_kb.json --model_name icelab/Cosmicroberta_KQA_pro`

# TRAINING
Files are already processed in training and validation split. 

Navigate into ./app folder

First train on original KQA Pro dataset

Note: --input_dir has to be adapted to model type (rob = RoBERTa) 

python -m Pretraining.train --input_dir ./kqa_processed_ent_rob --output_dir ./train_ent_cosmicrob --model_name_or_path icelab/cosmicroberta --val_batch_size=256 --train_batch_size=128 --learning_rate=1e-5 --save_steps=2400 --logging_steps=1200 --num_train_epochs=100 --eval_steps=1200 --wandb=0 --model_type roberta

### Pick a good checkpoint and fine-tune on DWQ dataset

python -m Pretraining.train --input_dir ./dwq_processed_rob --output_dir ./cosmicroberta_ft_dwq  --model_name_or_path icelab/Cosmicroberta_KQA_pro  --val_batch_size=256 --train_batch_size=128 --learning_rate=1e-5 --save_steps=120 --logging_steps=2 --eval_steps=60 --num_train_epochs=80 --wandb=0 --model_type roberta --crf_learning_rate 1e-4
  
### Evaluation (TBC)
For evaluation run: 

python -m Pretraining.eval --models_dir trained models directory --data_dir  ./dwq_processed_robthe 

# APP
To use the model for inference a simple implementation in streamlit is provided to answer question on the DISCOSDB extract. 

### Run locally 
*You have to have a local python environment installed (V3.8, and V3.7 tested)*

- Clone repository locally 
- Navigate with commandline tool to `./query_app/app`
- Install dependencies with `pip install -r ./requirements.txt`
  
    in respective folders  
- Steps for using SUTIME (getting time values from query): Download maven from https://maven.apache.org/download.cgi
- Follow installation steps on https://maven.apache.org/install.html (Set path variables accordingly )
- run Streamlit app with `streamlit run app.py -- --model_name <insert_model> --input_dir <dir_processed_files>` (standard settings are model from PaulD/IOA_ft_latest and processed files in ./dwq_processed_rob)
- A window should open in your internet browser, otherwise open an internet browser and type `localhost:8501` in address line

### Run with docker 
*You have to have docker installed on your machine*

- Clone repository locally
    in respective folders 

- Navigate in commandline tool to parent folder (with `docker-compose.yml file`)
- Make sure to run preprocessing in inference mode again for trained model 
- Open docker-compose.yml file with text editor and change entry after --model_name to your local model dir (standard settings are loading model from PaulD/IOA_ft_latest on Huggingface)
- run `docker compose up`
- Open an internet browser and type `localhost:8450` in address line to get to app



