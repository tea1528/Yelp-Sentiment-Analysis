import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import argparse
from models.RCNN import RCNN
from models.selfAttention import SelfAttention
from models.LSTM import LSTMClassifier
import yaml


print("-------Loading Configuration--------\n")
config_filepath = 'config.yaml'
with open(config_filepath) as f:
    config = yaml.load(f)

# define hyperparameters
learning_rate = config['training']['lr']
batch_size = config['training']['batch_size']
output_size = config['training']['output_size']
hidden_size = config['training']['hidden_size']
embedding_length = config['training']['emb_size']
epochs = config['training']['epochs']
device = config['training']['gpu_device']

device = torch.device(device if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# let the user identify checkpoint
argparser = argparse.ArgumentParser()
argparser.add_argument("--checkpoint", "-c", default=None, help="Provide a filepath to a checkpoint. Default is None")
args = argparser.parse_args()

# read text dataset
print("-------Loading Data--------\n")
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, label_map = load_data.load_dataset(embed_len=embedding_length, batch_size=batch_size)

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def load_checkpoint(fname, model, optim): 
    if os.path.isfile(fname):
        print("\nCheckpoint file found. Resuming from checkpoint.\n")
        checkpoint = torch.load(fname) 
        if 'model_state_dict' in checkpoint: 
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model parameters loaded from checkpoint.")
        if 'optimizer_state_dict' in checkpoint: 
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer parameters loaded from checkpoint.")
        if 'loss' in checkpoint: 
            min_loss = checkpoint['loss']
            print("Previous validation loss loaded.")
        if 'epoch' in checkpoint: 
            prev_epochs = checkpoint['epoch']
            print("Continuing training from epoch: {}".format(prev_epochs))

        return model, optim, min_loss, prev_epochs 
    
def train_model(model, train_iter, epoch, learning_rate):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda(device)
            
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.stars
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda(device)
            target = target.cuda(device)
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than the batch size.
            continue
        optim.zero_grad()
        prediction = model(text)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.stars
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda(device)
                target = target.cuda(device)
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	
# Define model
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
model.to(device)

prev_epochs = 0
# reload model if a checkpoint is specified
if args.checkpoint:
    model, optim, min_loss, prev_epochs = load_checkpoint(args.checkpoint, model, optim)

#model = torch.nn.DataParallel(model)
loss_fn = F.cross_entropy
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

for epoch in range(prev_epochs, epochs):
    train_loss, train_acc = train_model(model, train_iter, epoch, learning_rate)
    val_loss, val_acc = eval_model(model, valid_iter)
    scheduler.step(val_loss)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')
    print("Saving model for epoch {}...".format(epoch+1))
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss_fn
    }, 'checkpoint/'+'rcnn_ep{}.pth'.format(epoch))

print("Training complete with {} epochs.".format(epoch))

