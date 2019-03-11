import os
import argparse
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler.StepLR as StepLR 
import numpy as np
#from models.RCNN import RCNN
from models.selfAttention import SelfAttention

# define hyperparameters
learning_rate = 1e-3
batch_size = 128
output_size = 5
hidden_size = 100
embedding_length = 300
prev_epochs = 0 
epochs = 30
init_patience = 2 
lr_decay_rate = 0.1 



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model,loss_fn, train_iter, optim, epoch, learning_rate):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()

    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.stars
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 32.
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

def eval_model(model, loss_fn, val_iter):
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
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)

def load_checkpoint(fname, model, optim): 

	if os.path.isfile(fname):
		print("\nCheckpoint file found. Resuming from checkpoint.\n")
		checkpoint = torch.load(fname) 
		if 'model_state_dict' in checkpoint: 
			model.load_state_dict(checkpoint['model_state_dict'])
			print("Model parameters loaded from checkpoint.")
		if 'optimizer_state_dict' in checkpoint: 
			optim.load_state_dcit(checkpoint['optimizer_state_dict'])
			print("Optimizer parameters loaded from checkpoint.")
		if 'loss' in checkpoint: 
			min_loss = checkpoint['loss']
			print("Previous validation loss loaded.")
		if 'epoch' in checkpoint: 
			prev_epochs = checkpoint['epochs']
			print("Continuing training from epoch: {}".format(prev_epochs))

	return model, optim, min_loss, prev_epochs 


def main(): 

	# let the user identify checkpoint 
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--checkpoint", "-c", default=None, help="Provide a filepath to a checkpoint. Default is None")
	args = argparser.parse_args()

	# read text dataset
	TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(embed_len=embedding_length, batch_size=batch_size)

	# initiate model and optimizer
	model = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0)
 

	# reload model if a checkpoint is specified  
	if args.checkpoint: 
		model, optim, min_loss, prev_epochs = load_checkpoint(args.checkpoint)

	loss_fn = F.cross_entropy

	# training variable and scheduler initializiation 
	patience = init_patience 
	scheduler = StepLR(optim, step_size=1, gamma=lr_decay_rate)
	lr_reduced = False 
	min_loss = 1e20 

	for epoch in range(prev_epochs, epochs):
	    train_loss, train_acc = train_model(model,loss_fn, optim, train_iter, epoch, learning_rate)
	    val_loss, val_acc = eval_model(model, loss_fn, valid_iter)
	    
	    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

	    # reduce patience if current epoch's validation loss is greater than min validation loss 
	    if val_loss > min_loss: 
	    	patience -= 1 
	    	print("Validation loss increased. Patience reduced to: {}.".format(patience))

	    	if patience == 0:
	    		# early stop if validation loss does not decrease even with lr decay 
	    		if lr_reduced: 
		    		print("Patience  = 0 : Validation loss increased even with learning rate decay --> Early stopping invoked.")
		    		break 

		    	# decay lr 
	    		else: 
	    			print("Patience reached zero. Reducing learning rate by {}".format(lr_decay_rate))
	    			scheduler.step() 
	    			patience = init_patience 
	    			lr_reduced = True 
	    else: 
	    	# if val_loss goes down after lr is reduced, reset lr_reduced 
	    	lr_reduced = False

	    # min validation loss 
	    if val_loss < min_loss: 
	    	min_loss = val_loss 


	print("Training complete with {} epochs.".format(epoch))
	print("Saving model...")
	torch.save({
			'epoch': epoch, 
			'model_state_dict': model.state_dict(), 
			'optimizer_state_dict': optimizer.state_dict(), 
			'loss': loss
		}, 'rcnn_bs{}_lr{}.pth'.format(batch_size, learning_rate))



if __name__ == '__main__':
	main()
