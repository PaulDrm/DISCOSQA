from transformers import (BertConfig, BertModel, BertTokenizer, BertPreTrainedModel)
from .model import RelationPT
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, help='path to saved model checkpoints')
    parser.add_argument('--name', required=True, help='Name to save model with')
    args = parser.parse_args()
    name = args.name
    save_dir = args.save_dir#'/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-14399'
    #save_dir = '/content/drive/MyDrive/IOA/ProgramTransfer/models/checkpoint-76799'
    config_class, model_class, tokenizer_class = (BertConfig, RelationPT, BertTokenizer)
    print("load ckpt from {}".format(save_dir))
    config = config_class.from_pretrained(save_dir)#, num_labels = len(label_list))
    model = model_class.from_pretrained(save_dir, config = config)
    checkpoint = save_dir.split('\\')[-1]
    model.push_to_hub(f"{name}-{checkpoint}", use_temp_dir=True)#, organization="my-awesome-org")

if __name__ == '__main__':
    main()