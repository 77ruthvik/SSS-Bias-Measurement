from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import BertModel, BertTokenizer
import torch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kl_div
import cmath
import math
import gc
import re

from scipy.stats import wasserstein_distance

traits = ["sociable", "unsociable", "friendly", "unfriendly", "warm", "cold", "liked", "disliked", "outgoing", "shy", "moral", "immoral", "trustworthy", "untrustworthy", "sincere", "insincere", "fair", "unfair", "tolerant", "intolerant", "wealthy", "poor", "powerful", "powerless", "superior", "inferior", "influential", "uninfluential", "successful", "unsuccessful"]

#Get complex dynamic embedding of the target word given the WinoBias sentence
def get_targ_dyn(sentence, word, model_name, tokenizer, model):
  inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
  inp_ids = inputs['input_ids'].to("cuda")
  att_mask = inputs['attention_mask'].to("cuda")

  with torch.no_grad():
    out = model(input_ids=inp_ids, attention_mask=att_mask, output_attentions=True, output_hidden_states=True)

  last_hidden_states = out.hidden_states
  attention_weights = out.attentions

  tokenized_input = tokenizer.tokenize(sentence)
  target_tokens = tokenizer.tokenize(word)

  target_indices = []
  target_len = len(target_tokens)

  for i, token in enumerate(tokenized_input):
      #"▁"+token in target_tokens
        if(model_name!="ALBERT"):
            if (token in target_tokens):
                #print(i)
                target_indices.append(i)
                break
        else:
            if ("▁"+token in target_tokens):
                #print(i)
                target_indices.append(i)
                break
     

  if len(target_indices) == 0:
    print(sentence)
    print(word)
    print("Target word not found in sentence tokens.")
    placeholder_token = '.'  # You can use '<|endoftext|>' or define your own token
    if placeholder_token in tokenized_input:
        target_indices = [tokenized_input.index(placeholder_token)]

  word_embedding = torch.zeros(last_hidden_states[-1].shape[-1]).to("cuda")

  for layer_idx, (layer_attention, layer_hidden_state) in enumerate(zip(attention_weights, last_hidden_states[1:])):
      target_attention = layer_attention[0, :, target_indices, :].mean(dim=1)

      weighted_attention = target_attention.mean(dim=0)
      weighted_attention = weighted_attention / weighted_attention.sum()

      target_hidden_states = layer_hidden_state[0, target_indices, :]

      weighted_embedding = (target_hidden_states[0:1,:] * weighted_attention.unsqueeze(1)).sum(dim=0)

      word_embedding += weighted_embedding.squeeze(0)

  word_embedding /= len(attention_weights) #normalize
  word_embedding = word_embedding.cpu().numpy()

  SSN_targword = word_embedding/np.linalg.norm(word_embedding) #Normalize
  targ_phase = math.pi*SSN_targword

  targ_sup = SSN_targword*np.exp(1j*targ_phase)
  #outer_targ = np.outer(targ_sup, np.conj(targ_sup))

  return targ_sup

def compute_embeddings(sent, model_name, tokenizer, model):
  if(model_name == "GPT-2" or model_name=="GPT-2XL"):
    #GPT-2
    inp_ids = tokenizer(sent, return_tensors="pt").to("cuda")
    with torch.no_grad():
      out = model(**inp_ids)
    last_hidden_state = out.hidden_states[-1]
    simple_emb = last_hidden_state.mean(dim=1) #V_target
    simple_emb = simple_emb.cpu().numpy()

    return simple_emb
  else:
    #BERT, ALBERT, RoBERTa
    inp_ids = tokenizer(sent, return_tensors="pt", padding=True, truncation=True).to("cuda")
    with torch.no_grad():
      out = model(**inp_ids)
    last_hidden_state = out.last_hidden_state

    att_mask = inp_ids['attention_mask']
    mask_hid_state = last_hidden_state * att_mask.unsqueeze(-1)
    simple_emb = mask_hid_state.sum(dim=1) / att_mask.sum(dim=1, keepdim=True)

    simple_emb = simple_emb.cpu().numpy()

    return simple_emb

def plot(data, model_name):
    #Magnitude of Interference Plot
    plt.figure(figsize=(10, 6))
    labels = ['Male', 'Female']

    # Plot each set
    for idx, interference_values in enumerate(data):
        magnitudes = interference_values

        plt.scatter(traits, magnitudes, s=10, label=labels[idx])

    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.title('Absolute value of Interference plot of Male and Female Terms for '+model_name)
    plt.xlabel('SCM Traits')
    plt.ylabel('Interference')
    plt.ylim(bottom=0.09, top=0.14)  # Adjust the y-axis limits as needed
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

def get_SSS_wasserstein_distance(model_name, dataset_path):
    #Loading the pre-trained models
    if(model_name == "GPT-2"):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states="True").to("cuda")
        model.eval()
    elif(model_name == "GPT-2XL"):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        model = GPT2LMHeadModel.from_pretrained('results_GPT2XLChatGPT/checkpoint-1863', output_hidden_states="True").to("cuda")
        model.eval()
    elif(model_name == "RoBERTa"):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base', attn_implementation="eager").to("cuda")
        model.eval()
    elif(model_name == "ALBERT"):
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2', attn_implementation="eager").to("cuda")
        model.eval()
    elif(model_name == "BERT-base"):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to("cuda")
        model.eval()
    elif(model_name == "BERT-large"):
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased').to("cuda")
        model.eval()
    
    #Loading WinoBias datasets
    with open(dataset_path+'pro_stereotyped_type1.txt.test', 'r') as f:
        pro_sentences1 = f.readlines()

    cleaned_pro_sentences1 = [re.sub(r"^\d+\s*", "", sentence) for sentence in pro_sentences1]

    with open(dataset_path+'anti_stereotyped_type1.txt.test', 'r') as f:
        anti_sentences1 = f.readlines()
    
    cleaned_anti_sentences1 = [re.sub(r"^\d+\s*", "", sentence) for sentence in anti_sentences1]

    with open(dataset_path+'pro_stereotyped_type2.txt.test', 'r') as f:
        pro_sentences2 = f.readlines()

    cleaned_pro_sentences2 = [re.sub(r"^\d+\s*", "", sentence) for sentence in pro_sentences2]

    with open(dataset_path+'anti_stereotyped_type2.txt.test', 'r') as f:
        anti_sentences2 = f.readlines()

    cleaned_anti_sentences2 = [re.sub(r"^\d+\s*", "", sentence) for sentence in anti_sentences2]

    cleaned_pro_sentences = cleaned_pro_sentences1+cleaned_pro_sentences2
    cleaned_anti_sentences = cleaned_anti_sentences1+cleaned_anti_sentences2

    dyn_emb_target_pairs = {} #Dynamic Embeddings of Target Pairs
    targ_choices = ["[he]", "[she]", "[him]", "[her]", "[his]"]
    j = 0

    for i in range(0, len(cleaned_pro_sentences)):
        pro_sent = cleaned_pro_sentences[i]
        anti_sent = cleaned_anti_sentences[i]

        match_pro = [word for word in targ_choices if word in pro_sent][0] #Target word in Pro sentence
        match_anti = [word for word in targ_choices if word in anti_sent][0] #Target word in Anti sentence

        #Cleaning the target words
        match_pro = re.sub(r'\[([^\]]+)\]', r'\1', match_pro)
        match_anti = re.sub(r'\[([^\]]+)\]', r'\1', match_anti)

        #Check if no Target word is detected
        if(len(match_pro)==0 or len(match_anti)==0):
            print('Error')

        if(match_pro=="him" and match_anti=="him"):
            print('Ignore this sentence')
        else:
            if(match_pro=="he" or match_pro=="him" or match_pro=="his"):
                pro_dyn = get_targ_dyn(pro_sent, match_pro, model_name, tokenizer, model)
                anti_dyn = get_targ_dyn(anti_sent, match_anti, model_name, tokenizer, model)

                if("Male-Female" in dyn_emb_target_pairs):
                    dyn_emb_target_pairs["Male-Female"].append([pro_dyn, anti_dyn])
                else:
                    dyn_emb_target_pairs["Male-Female"] = [[pro_dyn, anti_dyn]]
            else:
                pro_dyn = get_targ_dyn(pro_sent, match_pro, model_name, tokenizer, model) #Female target
                anti_dyn = get_targ_dyn(anti_sent, match_anti, model_name, tokenizer, model) #Male target

                if("Male-Female" in dyn_emb_target_pairs):
                    dyn_emb_target_pairs["Male-Female"].append([anti_dyn, pro_dyn])
                else:
                    dyn_emb_target_pairs["Male-Female"] = [[anti_dyn, pro_dyn]]
    
    with open('SCM_generations_ChatGPT_10.json', 'r') as f:
        trait_sentences = json.load(f)
    
    trait_complex_emb = []
    trait_names = []
    j = 0
    temp = []

    for i in range(0, len(trait_sentences)):
        cur_trait = trait_sentences[i]['SCM']
        cur_sent = trait_sentences[i]['ChatGPT_generation']

        #Magnitude of ChatGPT sentence
        mag = compute_embeddings(cur_sent, model_name, tokenizer, model)

        norms = np.sum(mag ** 2, axis=1, keepdims=True) #np.linalg.norm(simple_emb, ord=2, axis=1, keepdims=True)
        SSN_v_targ = (mag/np.sqrt(norms)) #Computing the normalized embeddings

        #Phase
        phase = math.pi*SSN_v_targ

        #Complex vector
        comp_vec = SSN_v_targ*np.exp(1j*phase)
        trait_complex_emb.append(comp_vec)
        trait_names.append(cur_trait)
    
    scores = {} #Array of Wasserstein scores for all target pairs
    final_scores = {} #Average of Wasserstein scores for all target pairs
    final_inst = {}

    values_plot_women = []
    values_plot_men = []

    #Equation Variables
    alpha = 1/math.sqrt(2)
    beta = 1/math.sqrt(2)

    for key in dyn_emb_target_pairs.keys():
        #Iterating over all Target pairs
        targ_emb = dyn_emb_target_pairs[key]

        #Plot Variables
        overall_pro = []
        overall_anti = []

        inst = 0

        for pro_stereo, anti_stereo in targ_emb:
            #Computing the Spectrum for each pair of Pro-Stereotypical and Anti-Stereotypical sentence
            b = 0
            temp_pro = []
            temp_anti = []

            while(b<10):
                #Iterating over 10 instances of traits (powerful, powerless, ....)
                cur_dim_vec = [] #Current set of Complex vectors of Traits (size 30 (positive, negative, ..))

                for a in range(b, len(trait_complex_emb), 10):
                    cur_dim_vec.append(trait_complex_emb[a])

                b += 1

                #Pro Stereotype Spectrum
                pro_scores = []
                for j in range(0, len(cur_dim_vec), 2):
                    if(j+1 < len(cur_dim_vec)):
                        interference_pos = np.vdot(pro_stereo, np.squeeze(cur_dim_vec[j]))
                        interference_neg = np.vdot(pro_stereo, np.squeeze(cur_dim_vec[j+1]))

                        pro_scores.append(abs(interference_pos))
                        pro_scores.append(abs(interference_neg))

                #Anti Stereotype Spectrum
                anti_scores = []
                for j in range(0, len(cur_dim_vec), 2):
                    if(j+1 < len(cur_dim_vec)):
                        interference_pos = np.vdot(anti_stereo, np.squeeze(cur_dim_vec[j]))
                        interference_neg = np.vdot(anti_stereo, np.squeeze(cur_dim_vec[j+1]))

                        anti_scores.append(abs(interference_pos))
                        anti_scores.append(abs(interference_neg))

                temp_pro.append(pro_scores)
                temp_anti.append(anti_scores)

            mean_pro = np.mean(temp_pro, axis=0)
            mean_anti = np.mean(temp_anti, axis=0)

            overall_pro.append(mean_pro)
            overall_anti.append(mean_anti)

        values_plot_men.append(np.mean(overall_pro, axis=0))
        values_plot_women.append(np.mean(overall_anti, axis=0))
    
    data = [values_plot_men, values_plot_women]
    plot(data, model_name)

models = ["GPT-2", "GPT-2XL", "Llama", "RoBERTa", "ALBERT", "BERT-base", "BERT-large"]
get_SSS_wasserstein_distance(models[0], "WinoBias_TestData/")