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

def plot(overall_pro_type1, overall_anti_type1, overall_pro_type2, overall_anti_type2, model_name):
    #Magnitude of Interference Plot
    plt.figure(figsize=(10, 6))
    traits = ['sociable - unsociable', 'friendly - unfriendly', 'warm - cold', 'liked - disliked', 'outgoing - shy', 'moral - immoral', 'trustworthy - untrustworthy', 'sincere - insincere', 'fair - unfair', 'tolerant - intolerant', 'wealthy - poor', 'powerful - powerless', 'superior - inferior', 'influential - uninfluential', 'successful - unsuccessful']
    labels = ['Pro Type 1', 'Anti Type 1', 'Pro Type 2', 'Anti Type 2']
    data = [overall_pro_type1, overall_anti_type1, overall_pro_type2, overall_anti_type2]

    # Plot each set
    for idx, interference_values in enumerate(data):
        magnitudes = interference_values

        plt.scatter(traits, magnitudes, s=10, label=labels[idx])

    # Customize the plot
    plt.xticks(rotation=45, ha='right')
    plt.title('Magnitude Difference of Interference between Targets and Positive & Negative SCM traits for '+model_name)
    plt.xlabel('SCM Traits')
    plt.ylabel('Magnitude')
    plt.ylim(bottom=-0.025, top=0.02)  # Adjust the y-axis limits as needed
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

def get_accuracy_score(model_name, dataset_path):
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
    
    with open(dataset_path+'pro_stereotyped_type1.txt.test', 'r') as f:
        pro_sentences1 = f.readlines()
    
    cleaned_pro_sentences1 = [re.sub(r"^\d+\s*", "", sentence) for sentence in pro_sentences1]

    with open(dataset_path+'anti_stereotyped_type1.txt.test', 'r') as f:
        anti_sentences1 = f.readlines()
    
    cleaned_anti_sentences1 = [re.sub(r"^\d+\s*", "", sentence) for sentence in anti_sentences1]

    dyn_emb_target_pairs = {} #Dynamic Embeddings of Target Pairs
    
    targ_choices = ["[he]", "[she]", "[him]", "[her]", "[his]"]
    j = 0

    for i in range(0, len(cleaned_pro_sentences1)):
        pro_sent = cleaned_pro_sentences1[i]
        anti_sent = cleaned_anti_sentences1[i]

        match_pro = [word for word in targ_choices if word in pro_sent][0] #Target word in Pro sentence
        match_anti = [word for word in targ_choices if word in anti_sent][0] #Target word in Anti sentence

        #Cleaning the target words
        match_pro = re.sub(r'\[([^\]]+)\]', r'\1', match_pro)
        match_anti = re.sub(r'\[([^\]]+)\]', r'\1', match_anti)

        #Check if no Target word is detected
        if(len(match_pro)==0 or len(match_anti)==0):
            print('Error')

        #if(match_pro=="he" and match_anti=="her"):
            #print(pro_sent)
            #print(anti_sent)
        if(match_pro=="him" and match_anti=="him"):
            print('Ignore this sentence')
        else:
            pro_dyn = get_targ_dyn(pro_sent, match_pro, model_name, tokenizer, model)
            anti_dyn = get_targ_dyn(anti_sent, match_anti, model_name, tokenizer, model)

            if(match_pro+"-"+match_anti in dyn_emb_target_pairs):
                dyn_emb_target_pairs[match_pro+"-"+match_anti].append([pro_dyn, anti_dyn])
            else:
                dyn_emb_target_pairs[match_pro+"-"+match_anti] = [[pro_dyn, anti_dyn]]
    
    with open('SCM_generations_ChatGPT_10.json', 'r') as f:
        trait_sentences = json.load(f)
    
    trait_complex_emb = []
    trait_names = []
    j = 0
    temp = []

    for i in range(0, len(trait_sentences)):
        j += 1

        if(j<11):
            cur_trait = trait_sentences[i]['SCM']
            cur_sent = trait_sentences[i]['ChatGPT_generation']

            #Magnitude of ChatGPT sentence
            mag = compute_embeddings(cur_sent, model_name, tokenizer, model)
            temp.append(mag)

        if(j==10):
            #Averaging over the 10 sentences for each trait
            avg_emb = np.mean(temp, axis=0)

            #Normalizing
            norms = np.sum(avg_emb ** 2, axis=1, keepdims=True) #np.linalg.norm(simple_emb, ord=2, axis=1, keepdims=True)
            SSN_v_targ = (avg_emb/np.sqrt(norms)) #Computing the normalized embeddings

            #Phase
            phase = math.pi*SSN_v_targ

            #Complex vector
            comp_vec = SSN_v_targ*np.exp(1j*phase)
            trait_complex_emb.append(comp_vec)
            trait_names.append(cur_trait)
            temp = []
            j=0
    
    scores = {} #Array of Wasserstein scores for all target pairs
    final_scores = {} #Average of Wasserstein scores for all target pairs
    values_plot = []
    final_inst = {}

    #Equation Variables
    alpha = 1/math.sqrt(2)
    beta = 1/math.sqrt(2)
    inst = 0
    pro_sc = 0

    #Plot Variables
    overall_pro = []
    overall_anti = []

    temp_count = 0

    for key in dyn_emb_target_pairs.keys():
        #Iterating over all Target pairs
        targ_emb = dyn_emb_target_pairs[key]

        for pro_stereo, anti_stereo in targ_emb:
            #Computing the Spectrum for each pair of Pro-Stereotypical and Anti-Stereotypical sentence

            #Pro Stereotype Spectrum
            pro_scores_mag = [] #Pro sent - Interference magnitude differences between positive and negative traits
            pro_scores_angle = [] #Pro sent - Interference phase differences between positive and negative traits

            for j in range(0, len(trait_complex_emb), 2):
                if(j+1 < len(trait_complex_emb)):
                    interference_pos = np.vdot(pro_stereo, trait_complex_emb[j])
                    interference_neg = np.vdot(pro_stereo, trait_complex_emb[j+1])

                    mag_diff = np.abs(interference_pos) - np.abs(interference_neg)
                    ph_diff = np.angle(interference_pos) - np.angle(interference_neg)

                    pro_scores_mag.append(mag_diff)
                    pro_scores_angle.append(ph_diff)

            #Anti Stereotype Spectrum
            anti_scores_mag = [] #Anti sent - Interference magnitude differences between positive and negative traits
            anti_scores_angle = [] #Anti sent - Interference phase differences between positive and negative traits

            for j in range(0, len(trait_complex_emb), 2):
                if(j+1 < len(trait_complex_emb)):
                    interference_pos = np.vdot(anti_stereo, trait_complex_emb[j])
                    interference_neg = np.vdot(anti_stereo, trait_complex_emb[j+1])

                    mag_diff = np.abs(interference_pos) - np.abs(interference_neg)
                    ph_diff = np.angle(interference_pos) - np.angle(interference_neg)

                    anti_scores_mag.append(mag_diff)
                    anti_scores_angle.append(ph_diff)

            #Plots
            overall_pro.append(pro_scores_mag)
            overall_anti.append(anti_scores_mag)

            sum_pro_mag = np.sum(pro_scores_mag)
            sum_pro_angle = np.sum(pro_scores_angle)

            sum_anti_mag = np.sum(anti_scores_mag)
            sum_anti_angle = np.sum(anti_scores_angle)

            if(abs(sum_pro_mag) > abs(sum_anti_mag)): #and abs(sum_pro_angle) < abs(sum_anti_angle)
                pro_sc += 1

            inst += 1
    
    print("Accuracy of Type 1 Sentences: ")
    print(pro_sc/inst)

    overall_pro_type1 = np.mean(overall_pro, axis=0)
    overall_anti_type1 = np.mean(overall_anti, axis=0)

    #WinoBias Type 2 sentences
    with open(dataset_path+'pro_stereotyped_type2.txt.test', 'r') as f:
        pro_sentences2 = f.readlines()

    cleaned_pro_sentences2 = [re.sub(r"^\d+\s*", "", sentence) for sentence in pro_sentences2]

    with open(dataset_path+'anti_stereotyped_type2.txt.test', 'r') as f:
        anti_sentences2 = f.readlines()

    cleaned_anti_sentences2 = [re.sub(r"^\d+\s*", "", sentence) for sentence in anti_sentences2]

    dyn_emb_target_pairs = {} #Dynamic Embeddings of Target Pairs
    targ_choices = ["[he]", "[she]", "[him]", "[her]", "[his]"]
    j = 0

    for i in range(0, len(cleaned_pro_sentences2)):
        pro_sent = cleaned_pro_sentences2[i]
        anti_sent = cleaned_anti_sentences2[i]

        match_pro = [word for word in targ_choices if word in pro_sent][0] #Target word in Pro sentence
        match_anti = [word for word in targ_choices if word in anti_sent][0] #Target word in Anti sentence

        #Cleaning the target words
        match_pro = re.sub(r'\[([^\]]+)\]', r'\1', match_pro)
        match_anti = re.sub(r'\[([^\]]+)\]', r'\1', match_anti)

        #Check if no Target word is detected
        if(len(match_pro)==0 or len(match_anti)==0):
            print('Error')

        #if(match_pro=="he" and match_anti=="her"):
            #print(pro_sent)
            #print(anti_sent)
        if(match_pro=="him" and match_anti=="him"):
            print('Ignore this sentence')
        else:
            pro_dyn = get_targ_dyn(pro_sent, match_pro, model_name, tokenizer, model)
            anti_dyn = get_targ_dyn(anti_sent, match_anti, model_name, tokenizer, model)

            if(match_pro+"-"+match_anti in dyn_emb_target_pairs):
                dyn_emb_target_pairs[match_pro+"-"+match_anti].append([pro_dyn, anti_dyn])
            else:
                dyn_emb_target_pairs[match_pro+"-"+match_anti] = [[pro_dyn, anti_dyn]]

    scores = {} #Array of Wasserstein scores for all target pairs
    final_scores = {} #Average of Wasserstein scores for all target pairs
    values_plot = []
    final_inst = {}

    #Equation Variables
    alpha = 1/math.sqrt(2)
    beta = 1/math.sqrt(2)
    inst = 0
    pro_sc = 0

    #Plot Variables
    overall_pro = []
    overall_anti = []

    for key in dyn_emb_target_pairs.keys():
        #Iterating over all Target pairs
        targ_emb = dyn_emb_target_pairs[key]

        for pro_stereo, anti_stereo in targ_emb:
            #Computing the Spectrum for each pair of Pro-Stereotypical and Anti-Stereotypical sentence

            #Pro Stereotype Spectrum
            pro_scores_mag = [] #Pro sent - Interference magnitude differences between positive and negative traits
            pro_scores_angle = [] #Pro sent - Interference phase differences between positive and negative traits

            for j in range(0, len(trait_complex_emb), 2):
                if(j+1 < len(trait_complex_emb)):
                    interference_pos = np.vdot(pro_stereo, trait_complex_emb[j])
                    interference_neg = np.vdot(pro_stereo, trait_complex_emb[j+1])

                    mag_diff = np.abs(interference_pos) - np.abs(interference_neg)
                    ph_diff = np.angle(interference_pos) - np.angle(interference_neg)

                    pro_scores_mag.append(mag_diff)
                    pro_scores_angle.append(ph_diff)

            #Anti Stereotype Spectrum
            anti_scores_mag = [] #Anti sent - Interference magnitude differences between positive and negative traits
            anti_scores_angle = [] #Anti sent - Interference phase differences between positive and negative traits

            for j in range(0, len(trait_complex_emb), 2):
                if(j+1 < len(trait_complex_emb)):
                    interference_pos = np.vdot(anti_stereo, trait_complex_emb[j])
                    interference_neg = np.vdot(anti_stereo, trait_complex_emb[j+1])

                    mag_diff = np.abs(interference_pos) - np.abs(interference_neg)
                    ph_diff = np.angle(interference_pos) - np.angle(interference_neg)

                    anti_scores_mag.append(mag_diff)
                    anti_scores_angle.append(ph_diff)

            overall_pro.append(pro_scores_mag)
            overall_anti.append(anti_scores_mag)

            sum_pro_mag = np.sum(pro_scores_mag)
            sum_pro_angle = np.sum(pro_scores_angle)

            sum_anti_mag = np.sum(anti_scores_mag)
            sum_anti_angle = np.sum(anti_scores_angle)

            if(abs(sum_pro_mag) > abs(sum_anti_mag)): #and abs(sum_pro_angle) < abs(sum_anti_angle)
                pro_sc += 1

            inst += 1    

    #Accuracy
    print("Accuracy of Type 2 sentences: ")
    print(pro_sc/inst)

    overall_pro_type2 = np.mean(overall_pro, axis=0)
    overall_anti_type2 = np.mean(overall_anti, axis=0)

    #Plots
    plot(overall_pro_type1, overall_anti_type1, overall_pro_type2, overall_anti_type2, model_name)

models = ["GPT-2", "GPT-2XL", "Llama", "RoBERTa", "ALBERT", "BERT-base", "BERT-large"]
get_accuracy_score(models[0], "WinoBias_TestData/") #Change this line to run your preferred model

