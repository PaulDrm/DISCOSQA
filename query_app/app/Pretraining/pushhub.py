from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
#from .model_rob import RelationPT
import argparse
from Pretraining.model import RelationPT
from Pretraining.model_rob import RelationPT_rob
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
import pickle
from tqdm.notebook import tqdm
import torch
from transformers import (RobertaConfig, RobertaModel,AutoTokenizer, RobertaPreTrainedModel)
from huggingface_hub import HfApi



def load_classes(path):
  with open(os.path.abspath(path), 'rb') as f:
      input_ids = pickle.load(f)
      token_type_ids = pickle.load(f)
      attention_mask = pickle.load(f)
      # input_ids = torch.LongTensor(input_ids[:512,:]).to(device)
      # token_type_ids = torch.LongTensor(token_type_ids[:512,:]).to(device)
      # attention_mask = torch.LongTensor(attention_mask[:512,:]).to(device)
      input_ids = torch.LongTensor(input_ids)#.to(device)
      token_type_ids = torch.LongTensor(token_type_ids)#.to(device)
      attention_mask = torch.LongTensor(attention_mask)#.to(device)
  argument_inputs = {
    'input_ids': input_ids,
    'token_type_ids': token_type_ids,
    'attention_mask': attention_mask
  }
  return argument_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, help='path to saved model checkpoints')
    parser.add_argument('--name', required=True, help='Name to save model with')
    parser.add_argument('--input_dir', help= 'Name to input directory for entity files')
    parser.add_argument('--model_type', default="roberta")
    args = parser.parse_args()
    name = args.name
    save_dir = args.save_dir#'/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-14399'
    #save_dir = '/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-76799'
    #config_class, model_class, tokenizer_class = (BertConfig, RelationPT, BertTokenizer)
    if args.model_type == "roberta":
        print("load ckpt from {}".format(save_dir))
        config_class, model_class, tokenizer_class = (RobertaConfig, RelationPT_rob, AutoTokenizer)
        config = config_class.from_pretrained(save_dir)#, num_labels=len(label_list))
        #config.update({'vocab': vocab})
        tokenizer = tokenizer_class.from_pretrained('roberta-base', do_lower_case=False)
        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=False)
        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case = False)
        model = model_class.from_pretrained(save_dir, config=config)
        # checkpoint = save_dir.split('\\')[-1]
        checkpoint = save_dir.split("checkpoint")[1]
    else:
        config_class, model_class, tokenizer_class = (BertConfig, RelationPT, BertTokenizer)
        print("load ckpt from {}".format(save_dir))
        config = config_class.from_pretrained(save_dir)#, num_labels = len(label_list))
        #tokenizer = tokenizer_class.from_pretrained("bert-base-cased")
        tokenizer = tokenizer_class.from_pretrained("bert-base-cased", do_lower_case=False)
        model = model_class.from_pretrained(save_dir, config = config)
        # checkpoint = save_dir.split('\\')[-1]
        checkpoint = save_dir.split("checkpoint")[1]

    checkpoint= ""
    print("pushing tokenizer")
    tokenizer.push_to_hub(f"{name}{checkpoint}")
    print("pushing model")
    model.push_to_hub(f"{name}{checkpoint}", use_temp_dir=True)#, organization="my-awesome-org")

    api = HfApi()

    if torch.cuda.is_available():
        batch_num = 128
        argument_inputs = load_classes(args.input_dir + "/entity.pt", )
        #argument_inputs = load_classes(args.input_dir)
        data = TensorDataset(argument_inputs['input_ids'], argument_inputs['attention_mask'],
                             argument_inputs['token_type_ids'])
        data_sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=data_sampler, batch_size=batch_num)
        model.cuda()
        attribute_embeddings = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                # if i == 1:
                #    break
                inputs = batch[0].to(device)
                masks = batch[1].to(device)
                tags = batch[2].to(device)

                attribute_embeddings += model.bert(input_ids=inputs,
                                                   attention_mask=masks,
                                                   token_type_ids=tags)[1].cpu()
        attribute_embeddings = torch.stack(attribute_embeddings)
        with open(os.path.join(args.input_dir, 'entity_embeddings.pt'), 'wb') as f:
           pickle.dump(attribute_embeddings, f)

        api.upload_file(
            path_or_fileobj=args.input_dir + '/entity_embeddings.pt',
            path_in_repo="entity_embeddings.pt",
            repo_id=f"PaulD/{name}{checkpoint}",
            repo_type="model",
        )

if __name__ == '__main__':
    main()