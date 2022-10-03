import torch
from Pretraining.utils import *
from Pretraining.model import RelationPT
from Pretraining.data import DataLoader
from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)#*
from Pretraining.lr_scheduler import get_linear_schedule_with_warmup
from Pretraining.metric import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import torch.optim as optim

import wandb
#from IPython import embed
import pickle

os.environ["WANDB_API_KEY"] = "7ee09b0cec0f14411947fcf09144f4a72c09c411"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

def load_dataloader(path,vocab_json, batch_size):


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    vocab_json = os.path.join(args.input_dir, 'vocab.json')

    ## Load dataloaders
    relation_train_pt = os.path.join(args.input_dir, 'relation', 'train.pt')
    relation_val_pt = os.path.join(args.input_dir, 'relation', 'dev.pt')
    relation_train_loader = DataLoader(vocab_json, relation_train_pt, args.train_batch_size, training=True)
    relation_val_loader = DataLoader(vocab_json, relation_val_pt, args.val_batch_size)

    concept_train_pt = os.path.join(args.input_dir, 'concept', 'train.pt')
    concept_val_pt = os.path.join(args.input_dir, 'concept', 'dev.pt')
    concept_train_loader = DataLoader(vocab_json, concept_train_pt, args.train_batch_size, training=True)
    concept_val_loader = DataLoader(vocab_json, concept_val_pt, args.val_batch_size)

    entity_train_pt = os.path.join(args.input_dir, 'entity', 'train.pt')
    entity_val_pt = os.path.join(args.input_dir, 'entity', 'dev.pt')
    entity_train_loader = DataLoader(vocab_json, entity_train_pt, args.train_batch_size, training=True)
    entity_val_loader = DataLoader(vocab_json, entity_val_pt, args.val_batch_size)