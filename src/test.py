import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import load_data
import pandas as pd
import torchtext
import os
import torch.optim as optim
from models.RCNN import RCNN
import yaml


print("-------Loading Configuration--------\n")
config_filepath = 'config.yaml'
with open(config_filepath) as f:
    config = yaml.load(f)

# define hyperparamters
batch_size = config['testing']['batch_size']
output_size = config['testing']['output_size']
hidden_size = config['testing']['hidden_size']
embedding_length = config['testing']['emb_size']
device = config['testing']['gpu_device']
saved = config['testing']['saved_path']
val = pd.read_csv(config['testing']['valid_path'])
test = pd.read_csv(config['testing']['test_path'])

device = torch.device(device if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


# read text dataset
print("-------Loading Data--------\n")
TEXT, vocab_size, word_embeddings, _, _, mapping = load_data.load_dataset(embed_len=embedding_length, batch_size=batch_size)

# Map the output of the model (0-4) back to original region (1-5)
def map_reverse(label):
    for k, v in mapping.items():
        if label == v:
            return float(k)

# Load the saved model
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
print(f"\nLoading checkpoint file: {saved}\n")
checkpoint = torch.load(saved)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# Run inference on validation set
pred_val = []
for i in range(len(val['text'])):
    with torch.no_grad():
        test_sen = TEXT.preprocess(val['text'][i])
        test_sen = np.asarray([[TEXT.vocab.stoi[x] for x in test_sen]])
        test_sen = torch.LongTensor(test_sen)
        test_tensor = Variable(test_sen)
        test_tensor = test_tensor.cuda(device)
        model.eval()
        output = model(test_tensor, 1)
        out = F.softmax(output, 1)
        out = torch.argmax(out[0]).item()
        out = map_reverse(out)
        pred_val.append(out)

print(f'The validation accuracy: {np.mean(pred_val == val["stars"])}')

pred = []
for i in range(len(test['text'])):
    with torch.no_grad():
        test_sen = TEXT.preprocess(test['text'][i])
        test_sen = np.asarray([[TEXT.vocab.stoi[x] for x in test_sen]])
        test_sen = torch.LongTensor(test_sen)
        test_tensor = Variable(test_sen)
        test_tensor = test_tensor.cuda(device)
        model.eval()
        output = model(test_tensor, 1)
        out = F.softmax(output, 1)
        out = torch.argmax(out[0]).item()

        out = map_reverse(out)
        pred.append(out)

# This is the accuracy of our model on test set
print(f'The test accuracy: {np.mean(pred == test["stars"])}')

print("Saving prediction...")

sub_df = pd.DataFrame()
sub_df["pre"] = pred
sub_df.to_csv("results/pre.csv", index=False)

