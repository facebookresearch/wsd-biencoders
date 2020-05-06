'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import torch
from torch.nn import functional as F
import os
import sys
import copy
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *

import numpy as np
import random

from wsd_models.util import *

parser = argparse.ArgumentParser(description='BERT Frozen Probing Model for WSD')
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--ckpt', type=str, required=True,
	help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--kshot', type=int, default=-1,
	help='if set to k (1+), will filter training data to only have up to k examples per sense')
parser.add_argument('--data-path', type=str, required=True,
	help='Location of top-level directory for the Unified WSD Framework')

parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
	help='Which evaluation split on which to evaluate probe')

def wn_keys(data):
	keys = []
	for sent in data:
		for form, lemma, pos, inst, _ in sent:
			if inst != -1:
				key = generate_key(lemma, pos)
				keys.append(key)
	return keys

def batchify(data, bsz=1):
	print('Batching data with bsz={}...'.format(bsz))
	batched_data = []
	for i in range(0, len(data), bsz):
		if i+bsz < len(data): d_arr = data[i:i+bsz]
		else: d_arr = data[i:] #get remainder examples 
		batched_ids = torch.cat([ids for ids, _, _, _ in d_arr], dim=0)
		batched_masks = torch.stack([mask for _, mask, _, _ in d_arr], dim=0)
		batched_insts = [inst for _, _, inst, _ in d_arr]
		batched_labels = torch.cat([label for _, _, _, label in d_arr], dim=0)
		batched_data.append((batched_ids, batched_masks, batched_insts, batched_labels))
	return batched_data

#takes in text data, tensorizes it for BERT, runs though BERT,
#filters out the context words (not labeled), and averages
#the representation(s) for words/phrases to be disambiguated
#output is tuples of (input tensor prepared for linear probing model, 
#instance numbers (for dataset), tensor of label indexes)
def preprocess(tokenizer, context_model, text_data, label_space, label_map):
	processed_examples = []
	output_masks = []
	instances = []
	label_indexes = []

	#tensorize data
	for sent in tqdm(text_data):
		sent_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])] #aka sos token, returns a list with single index
		bert_mask = [-1]
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			word_ids = torch.tensor([tokenizer.encode(word.lower())])
			sent_ids.append(word_ids)

			if inst != -1:
				#masking for averaging of bert outputs
				bert_mask.extend([idx]*word_ids.size(-1))

				#tracking instance for sense-labeled word
				instances.append(inst)

				#adding label tensor for senes-labeled word
				if label in label_space:
					label_indexes.append(torch.tensor([label_space.index(label)]))
				else:
					label_indexes.append(torch.tensor([label_space.index('n/a')]))

				#adding appropriate label space for sense-labeled word (we only use this for wsd task)
				key = generate_key(lemma, pos)
				if key in label_map:
					l_space = label_map[key]
					o_mask = torch.zeros(len(label_space))
					for l in l_space: o_mask[l] = 1
					output_masks.append(o_mask)
				else:
					output_masks.append(torch.ones(len(label_space))) #let this predict whatever -- should not use this (default to backoff for unseen forms)

			else:
				bert_mask.extend([-1]*word_ids.size(-1))

		#add eos token
		sent_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)])) #aka eos token
		bert_mask.append(-1)
			
		sent_ids = torch.cat(sent_ids, dim=-1)

		#run inputs through frozen bert
		sent_ids = sent_ids.cuda()

		with torch.no_grad(): 
			output = context_model(sent_ids)[0].squeeze().cpu()

		#average outputs for subword units in same word/phrase, drop unlabeled words	
		combined_outputs = process_encoder_outputs(output, bert_mask)
		processed_examples.extend(combined_outputs)

	#package preprocessed data together + return
	data = list(zip(processed_examples, output_masks, instances, label_indexes))
	return data

def _train(train_data, probe, optim, criterion, bsz=1, silent=False):
	if not silent: train_data = tqdm(train_data)
	for input_ids, output_mask, _, label in train_data:
		input_ids = input_ids.cuda()
		output_mask = output_mask.cuda()
		label = label.cuda()

		optim.zero_grad()
		output = probe(input_ids)
		#mask to candidate senses for target word
		output = torch.mul(output, output_mask)
		#set masked out items to -inf to get proper probabilities over the candidate senses
		output[output == 0] = float('-inf')

		output = F.softmax(output, dim=-1)
		loss = criterion(output, label)

		batch_sz = loss.size(0)
		loss = loss.sum()/batch_sz
		loss.backward()
		optim.step()

	return probe, optim

def _eval(eval_data, probe, label_space):
	eval_preds = []
	for input_ids, output_mask, inst, _ in eval_data:
		input_ids = input_ids.cuda()
		output_mask = output_mask.cuda()

		#run example through model
		with torch.no_grad(): 
			output = probe(input_ids)
			#mask to candidate senses for target word
			output = torch.mul(output, output_mask)
			#set masked out items to -inf to get proper probabilities over the candidate senses
			output[output == 0] = float('-inf')
			output = F.softmax(output, dim=-1)

		#get predicted label
		pred_id = output.topk(1, dim=-1)[1].squeeze().item()
		pred_label = label_space[pred_id]
		eval_preds.append((inst[0], pred_label))
	return eval_preds

def _eval_with_backoff(eval_data, probe, label_space, wn_senses, coverage, keys):
	eval_preds = []
	for key, (input_ids, output_mask, inst, _) in zip(keys, eval_data):
		input_ids = input_ids.cuda()
		output_mask = output_mask.cuda()

		if key in coverage:
			#run example through model
			with torch.no_grad(): 
				output = probe(input_ids)
				output = torch.mul(output, output_mask)
				#set masked out items to -inf to get proper probabilities over the candidate senses
				output[output == 0] = float('-inf')
				
				output = F.softmax(output, dim=-1)
			#get predicted label
			pred_id = output.topk(1, dim=-1)[1].squeeze().item()
			pred_label = label_space[pred_id]
			eval_preds.append((inst[0], pred_label))
		#backoff to wsd for lemma+pos
		else:
			#this is ws1 for given key
			pred_label = wn_senses[key][0]
			eval_preds.append((inst[0], pred_label))

	return eval_preds

def train_probe(args):
	lr = args.lr
	bsz = args.bsz

	#create passed in ckpt dir if doesn't exist
	if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

	'''
	LOAD PRETRAINED BERT MODEL 
	'''

	#model loading code based on pytorch_transformers README example
	tokenizer = load_tokenizer(args.encoder_name)
	pretrained_model, output_dim = load_pretrained_model(args.encoder_name)
	pretrained_model = pretrained_model.cuda()

	'''
	LOADING IN TRAINING AND EVAL DATA
	'''
	print('Loading data + preprocessing...')
	sys.stdout.flush()
	#loading WSD (semcor) data + convert to supersenses
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')

	#filter train data for k-shot learning
	if args.kshot > 0: 
		train_data = filter_k_examples(train_data, args.kshot)
	
	task_labels, label_map = get_label_space(train_data)
	print('num labels = {} + 1 unknown label'.format(len(task_labels)-1))

	train_data = preprocess(tokenizer, pretrained_model, train_data, task_labels, label_map)
	train_data = batchify(train_data, bsz=args.bsz)

	num_epochs = args.epochs
	if args.kshot > 0:
		NUM_STEPS = 176600 #hard coded for fair comparision with full model on default num. of epochs
		num_batches = len(train_data)
		num_epochs = NUM_STEPS//num_batches #recalculate number of epochs
		overflow_steps = NUM_STEPS%num_batches #num steps in last overflow epoch (if there is one, otherwise 0)
		t_total = NUM_STEPS #manually set number of steps for lr schedule
		if overflow_steps > 0: num_epochs+=1 #add extra epoch for overflow steps
		print('Overriding args.epochs and training for {} epochs...'.format(epochs))

	#loading eval data & convert to supersense tags
	#dev set = semeval2007
	semeval2007_path = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/')
	semeval2007_data = load_data(semeval2007_path, 'semeval2007')
	semeval2007_data = preprocess(tokenizer, pretrained_model, semeval2007_data, task_labels, label_map)
	semeval2007_data = batchify(semeval2007_data, bsz=1)

	''' 
	SET UP PROBING MODEL FOR TASK
	'''

	#probing model = projection layer to label space, loss function, and optimizer
	probe = torch.nn.Linear(output_dim, len(task_labels))
	probe = probe.cuda()
	criterion = torch.nn.CrossEntropyLoss(reduction='none')
	optim = torch.optim.Adam(probe.parameters(), lr=lr)

	'''
	TRAIN PROBING MODEL ON SEMCOR DATA
	'''

	best_dev_f1 = 0.
	print('Training probe...')
	sys.stdout.flush()
	for epoch in range(1, num_epochs+1):
		#train on full dataset
		probe_optim = _train(train_data, probe, optim, criterion, bsz=bsz, silent=args.silent)

		#eval probe on dev set (semeval2007)
		eval_preds = _eval(semeval2007_data, probe, task_labels)

		#generate predictions file
		pred_filepath = os.path.join(args.ckpt, 'tmp_predictions.txt')
		with open(pred_filepath, 'w') as f:
			for inst, prediction in eval_preds:
				f.write('{} {}\n'.format(inst, prediction))

		#run predictions through scorer
		gold_filepath = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/semeval2007.gold.key.txt')
		scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
		_, _, dev_f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
		print('Dev f1 after {} epochs = {}'.format(epoch, dev_f1))
		sys.stdout.flush() 

		if dev_f1 >= best_dev_f1:
			print('updating best model at epoch {}...'.format(epoch))
			sys.stdout.flush() 
			best_dev_f1 = dev_f1
			#save to file if best probe so far on dev set
			probe_fname = os.path.join(args.ckpt, 'best_model.ckpt')
			with open(probe_fname, 'wb') as f:
				torch.save(probe.state_dict(), f)
		sys.stdout.flush()

		#shuffle train data after every epoch
		random.shuffle(train_data)

	return

def evaluate_probe(args):
	print('Evaluating WSD probe on {}...'.format(args.split))

	'''
	LOAD TOKENIZER + BERT MODEL
	'''
	tokenizer = load_tokenizer(args.encoder_name)
	pretrained_model, output_dim = load_pretrained_model(args.encoder_name)
	pretrained_model = pretrained_model.cuda()

	'''
	GET LABEL SPACE
	'''
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')
	task_labels, label_map = get_label_space(train_data)
	#for backoff eval
	train_keys = wn_keys(train_data)
	coverage = set(train_keys)

	'''
	LOAD TRAINED PROBE
	'''
	probe = torch.nn.Linear(output_dim, len(task_labels))
	probe_path = os.path.join(args.ckpt, 'best_model.ckpt')
	probe.load_state_dict(torch.load(probe_path))
	probe = probe.cuda()

	'''
	LOAD EVAL SET
	'''
	eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(args.split))
	eval_data = load_data(eval_path, args.split)
	#for backoff
	eval_keys = wn_keys(eval_data)
	eval_data = preprocess(tokenizer, pretrained_model, eval_data, task_labels, label_map)
	eval_data = batchify(eval_data, bsz=1)

	'''
	EVALUATE PROBE w/o backoff
	'''
	eval_preds = _eval(eval_data, probe, task_labels)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('f1 of WSD probe on {} test set = {}'.format(args.split, f1))

	'''
	EVALUATE PROBE with backoff
	'''
	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)
	eval_preds = _eval_with_backoff(eval_data, probe, task_labels, wn_senses, coverage, eval_keys)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_backoff_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('f1 of BERT probe (with backoff) = {}'.format(f1))

	return

if __name__ == "__main__":
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...")
		quit()

	args = parser.parse_args()
	print(args) 

	#set random seeds
	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)   
	
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic=True

	if args.eval: 
		evaluate_probe(args)
	else: 
		train_probe(args)


