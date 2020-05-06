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
import math
import copy
import argparse
from tqdm import tqdm
import pickle
from pytorch_transformers import *

import random
import numpy as np

from wsd_models.util import *
from wsd_models.models import PretrainedClassifier

parser = argparse.ArgumentParser(description='Finetuning Pretrained Encoders for WSD')
parser.add_argument('--rand_seed', type=int, default=42)
parser.add_argument('--grad-norm', type=float, default=1.0)
parser.add_argument('--silent', action='store_true',
	help='Flag to supress training progress bar for each epoch')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--warmup', type=int, default=2000)
parser.add_argument('--max-length', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--bsz', type=int, default=8)
parser.add_argument('--encoder-name', type=str, default='bert-base',
	choices=['bert-base', 'bert-large', 'roberta-base', 'roberta-large'])
parser.add_argument('--ckpt', type=str, required=True,
	help='filepath at which to save best probing model (on dev set)')
parser.add_argument('--proj-ckpt', type=str, default='',
	help='filepath to a pretrained projection layer/probe (trained with frozen_pretrained_model.py) to optionally use that as projection layer initalization')
parser.add_argument('--data-path', type=str, required=True,
	help='Location of top-level directory for the Unified WSD Framework')

parser.add_argument('--eval', action='store_true',
	help='Flag to set script to evaluate probe (rather than train)')
parser.add_argument('--split', type=str, default='semeval2007',
	choices=['semeval2007', 'senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'all-test'],
	help='Which evaluation split on which to evaluate probe')

#updated to organize keys by sentence to work with pretrained model
def wn_keys(data):
	keys = []
	for sent in data:
		sent_keys = []
		for form, lemma, pos, inst, _ in sent:
			if inst != -1:
				key = generate_key(lemma, pos)
				sent_keys.append(key)
		if len(sent_keys) > 0: keys.append(sent_keys)
	return keys

#takes in text data and indexes it for pretrained encoder + batching
#updated to return data organized by sentence to work with pretrained model
def preprocess(tokenizer, text_data, label_space, label_map, bsz=1, max_len=-1):
	if max_len == -1: 
		assert bsz==1 #otherwise need max_len for padding

	input_ids = []
	input_masks = []
	bert_masks = []
	output_masks = []
	instances = []
	label_indexes = []

	#tensorize data
	for sent in text_data:
		sent_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])] #cls token aka sos token, returns a list with index
		b_masks = [-1]
		o_masks = []
		sent_insts = []
		sent_labels = []

		ex_count = 0 #DEBUGGING
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
			sent_ids.extend(word_ids)

			if inst != -1:
				ex_count += 1 #DEBUGGING
				b_masks.extend([idx]*len(word_ids))

				sent_insts.append(inst)
				if label in label_space:
					sent_labels.append(torch.tensor([label_space.index(label)]))
				else:
					sent_labels.append(torch.tensor([label_space.index('n/a')]))

				#adding appropriate label space for sense-labeled word (we only use this for wsd task)
				key = generate_key(lemma, pos)
				if key in label_map:
					l_space = label_map[key]
					mask = torch.zeros(len(label_space))
					for l in l_space: mask[l] = 1
					o_masks.append(mask)
				else:
					o_masks.append(torch.ones(len(label_space))) #let this predict whatever -- should not use this (default to backoff for unseen forms)

			else:
				b_masks.extend([-1]*len(word_ids))

			#break if we reach max len so we don't keep overflowing examples
			if max_len != -1 and len(sent_ids) >= (max_len-1):
				break

		#pad inputs + add eos token
		sent_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)])) #aka eos token
		input_mask = [1]*len(sent_ids)
		b_masks.append(-1)
		sent_ids, input_mask, b_masks = normalize_length(sent_ids, input_mask, b_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

		#not including examples sentences with no annotated sense data	
		if len(sent_insts) > 0:
			input_ids.append(torch.cat(sent_ids, dim=-1))
			input_masks.append(torch.tensor(input_mask).unsqueeze(dim=0))
			bert_masks.append(torch.tensor(b_masks).unsqueeze(dim=0))
			output_masks.append(torch.stack(o_masks, dim=0))
			instances.append(sent_insts)
			label_indexes.append(torch.cat(sent_labels, dim=0))

	#batch data now that we pad it
	data = list(zip(input_ids, input_masks, bert_masks, output_masks, instances, label_indexes))
	if bsz > 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			input_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
			input_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
			bert_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
			output_mask = torch.cat([x for _,_,_,x,_,_ in b], dim=0)
			instances = []
			for _,_,_,_,x,_ in b: instances.extend(x)
			labels = torch.cat([x for _,_,_,_,_,x in b], dim=0)
			batched_data.append((input_ids, input_mask, bert_mask, output_mask, instances, labels))
		return batched_data
	else:  return data

def _train(train_data, model, optim, schedule, criterion, max_grad_norm=1.0, silent=False):
	model.train()
	total_loss = 0.

	if not silent: train_data = tqdm(train_data)
	for input_ids, input_mask, bert_mask, output_mask, _, label in train_data:
		input_ids = input_ids.cuda()
		input_mask = input_mask.cuda()
		output_mask = output_mask.cuda()
		label = label.cuda()

		model.zero_grad()
		output = model.forward(input_ids, input_mask, bert_mask)
		#mask output to appropriate senses for target word
		output = torch.mul(output, output_mask)
		#set masked out items to -inf to get proper probabilities over the candidate senses
		output[output == 0] = float('-inf')

		output = F.softmax(output, dim=-1)

		loss = criterion(output, label)
		total_loss += loss.sum().item()
		loss_sz = loss.size(0)
		loss=loss.sum()/loss_sz
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
		optim.step()
		schedule.step() # Update learning rate schedule

	return model, optim, schedule, total_loss

def _eval(eval_data, model, label_space):
	model.eval()
	eval_preds = []
	for input_ids, input_mask, bert_mask, output_mask, insts, labels in eval_data:
		input_ids = input_ids.cuda()
		input_mask = input_mask.cuda()
		output_mask = output_mask.cuda()
		labels = labels.cuda()

		#run example through model
		with torch.no_grad(): 
			outputs = model.forward(input_ids, input_mask, bert_mask)
			#mask to candidate senses for target word
			outputs = torch.mul(outputs, output_mask)
			#set masked out items to -inf to get proper probabilities over the candidate senses
			outputs[outputs == 0] = float('-inf')

			outputs = F.softmax(outputs, dim=-1)

		for i, output in enumerate(outputs):
			inst = insts[i]
			#get predicted label
			pred_id = output.topk(1, dim=-1)[1].squeeze().item()
			pred_label = label_space[pred_id]
			eval_preds.append((inst, pred_label))

	return eval_preds

def _eval_with_backoff(eval_data, model, label_space, wn_senses, coverage, keys):
	model.eval()
	eval_preds = []

	for sent_keys, (input_ids, input_mask, bert_mask, output_mask, insts, _) in zip(keys, eval_data):
		input_ids = input_ids.cuda()
		input_mask = input_mask.cuda()
		output_mask = output_mask.cuda()
		#run example through model
		with torch.no_grad(): 
			outputs = model.forward(input_ids, input_mask, bert_mask)
			#mask to candidate senses for target word
			outputs = torch.mul(outputs, output_mask)
			#set masked out items to -inf to get proper probabilities over the candidate senses
			outputs[outputs == 0] = float('-inf')
				
			outputs = F.softmax(outputs, dim=-1)

		for i, output in enumerate(outputs):
			k = sent_keys[i]
			inst = insts[i]
			if k in coverage:
				#get predicted label
				pred_id = output.topk(1, dim=-1)[1].squeeze().item()
				pred_label = label_space[pred_id]
				eval_preds.append((inst, pred_label))

			#backoff to wsd for lemma+pos
			else:
				#this is ws1 for given key
				pred_label = wn_senses[k][0]
				eval_preds.append((inst, pred_label))

	return eval_preds

def train_model(args):
	print('Finetuning pretrained model on WSD...')
	#create passed in ckpt dir if doesn't exist
	if not os.path.exists(args.ckpt): os.mkdir(args.ckpt)

	'''
	LOAD PRETRAINED MODEL'S TOKENIZER 
	'''
	#model loading code based on pytorch_transformers README example
	tokenizer = load_tokenizer(args.encoder_name)

	'''
	LOADING IN TRAINING AND EVAL DATA
	'''
	print('Loading data + preprocessing...')
	sys.stdout.flush()
	#loading WSD (semcor) data + convert to supersenses
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')
	
	#calculate label space
	label_space, label_map = get_label_space(train_data)
	print('num labels = {} + 1 unknown label'.format(len(label_space)-1))

	train_data = preprocess(tokenizer, train_data, label_space, label_map, bsz=args.bsz, max_len=args.max_length)

	#dev set = semeval2007
	semeval2007_path = os.path.join(args.data_path, 'Evaluation_Datasets/semeval2007/')
	semeval2007_data = load_data(semeval2007_path, 'semeval2007')
	semeval2007_data = preprocess(tokenizer, semeval2007_data, label_space, label_map, bsz=1, max_len=-1)

	''' 
	SET UP FINETUNING MODEL, OPTIMIZER, AND LR SCHEDULE
	'''

	model = PretrainedClassifier(len(label_space), args.encoder_name, args.proj_ckpt)
	if args.multigpu: model = torch.nn.DataParallel(model)
	model = model.cuda()

	criterion = torch.nn.CrossEntropyLoss(reduction='none')

	#optimize + scheduler from pytorch_transformers package
	#this is from pytorch_transformers finetuning code
	weight_decay = 0.0 #this could be a parameter
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
	adam_epsilon = 1e-8
	t_total = len(train_data)*args.epochs
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=adam_epsilon)
	schedule = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup, t_total=t_total)

	'''
	TRAIN FINETUNING MODEL ON SEMCOR DATA
	'''

	best_dev_f1 = 0.
	print('Training probe...')
	sys.stdout.flush()

	for epoch in range(1, args.epochs+1):
		#train on full dataset

		model, optimizer, schedule, train_loss = _train(train_data, model, optimizer, schedule, criterion, max_grad_norm=args.grad_norm, silent=args.silent)
		#eval probe on dev set (semeval2007)
		eval_preds = _eval(semeval2007_data, model, label_space)

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
			model_fname = os.path.join(args.ckpt, 'best_model.ckpt')
			with open(model_fname, 'wb') as f:
				torch.save(model.state_dict(), f)
			sys.stdout.flush()

		#shuffle train data after every epoch
		random.shuffle(train_data)


	return

def evaluate_model(args):
	print('Evaluating model on {} for WSD...'.format(args.split))

	'''
	LOAD LABEL SPACE
	'''
	train_path = os.path.join(args.data_path, 'Training_Corpora/SemCor/')
	train_data = load_data(train_path, 'semcor')
	coverage = set([k for keys in wn_keys(train_data) for k in keys])
	task_labels, label_map = get_label_space(train_data)

	'''
	LOAD TRAINED MODEL
	'''
	model = PretrainedClassifier(len(task_labels), args.encoder_name, '')
	model_path = os.path.join(args.ckpt, 'best_model.ckpt')
	model.load_state_dict(torch.load(model_path))
	model = model.cuda()

	'''
	LOAD TOKENIZER
	'''
	tokenizer = load_tokenizer(args.encoder_name)

	'''
	LOAD EVAL SET
	'''
	eval_path = os.path.join(args.data_path, 'Evaluation_Datasets/{}/'.format(args.split))
	eval_data = load_data(eval_path, args.split)
	#get keys to perform evaluation with backoff 
	eval_keys = wn_keys(eval_data)
	eval_data = preprocess(tokenizer, eval_data, task_labels, label_map, bsz=1, max_len=-1)

	'''
	EVALUATE MODEL w/o backoff
	'''
	eval_preds= _eval(eval_data, model, task_labels)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('F1 on {} test set = {}'.format(args.split, f1))

	wn_path = os.path.join(args.data_path, 'Data_Validation/candidatesWN30.txt')
	wn_senses = load_wn_senses(wn_path)
	eval_preds = _eval_with_backoff(eval_data, model, task_labels, wn_senses, coverage, eval_keys)

	#generate predictions file
	pred_filepath = os.path.join(args.ckpt, './{}_backoff_predictions.txt'.format(args.split))
	with open(pred_filepath, 'w') as f:
		for inst, prediction in eval_preds:
			f.write('{} {}\n'.format(inst, prediction))

	#run predictions through scorer
	gold_filepath = os.path.join(eval_path, '{}.gold.key.txt'.format(args.split))
	scorer_path = os.path.join(args.data_path, 'Evaluation_Datasets')
	p, r, f1 = evaluate_output(scorer_path, gold_filepath, pred_filepath)
	print('F1 (with backoff) = {}'.format(f1))

	return

if __name__ == "__main__":
	if not torch.cuda.is_available():
		print("Need available GPU(s) to run this model...")
		quit()

	#parse args
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

	#evaluate model saved at checkpoint or...
	if args.eval: evaluate_model(args)
	#finetune pretrained model
	else: train_model(args)


