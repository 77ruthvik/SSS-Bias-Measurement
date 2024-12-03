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
import ast

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

def get_SSS_wasserstein_distance(model_name, dataset_path, dataset_name):
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
    
    if(dataset_name == "Winobias"):
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
    elif(dataset_name=="Crows-pairs"):
        df = pd.read_csv(dataset_path+'crows_pairs_anonymized.csv')
        data_df = df[(df['stereo_antistereo']=="stereo") & (df['bias_type']=="gender")]
        cleaned_pro_sentences = data_df['sent_more'].tolist()
        cleaned_anti_sentences = data_df['sent_less'].tolist()

        dyn_emb_target_pairs = {} #Dynamic Embeddings of Target Pairs
        targ_choices = ["woman", "Women", "Woman", "women", "daughter", "son", "female", "Girls", "girls", "Boys", "boys", "girl", "Girl", "male", "uncle", "aunt", "man", "she", "She", "men", "Men", "His", "his", "Boy", "boy", "mom", "Mom", "Dad", "dad", "her", "Her", "He", "he"]
        targ_choices.sort(key=str.lower, reverse=True)
        j = 0

        for i in range(0, len(cleaned_pro_sentences)):
            pro_sent = cleaned_pro_sentences[i]
            anti_sent = cleaned_anti_sentences[i]

            match_pro = [word for word in targ_choices if re.search(r'\b' + re.escape(word) + r'\b', pro_sent)] #Target word in Pro sentence
            match_anti = [word for word in targ_choices if re.search(r'\b' + re.escape(word) + r'\b', anti_sent)] #Target word in Anti sentence

            #Check if no Target word is detected
            if(len(match_pro)==0 or len(match_anti)==0):
                j += 1
            else:
                if(match_pro[0]==match_anti[0]):
                    j += 1
                else:
                    match_pro = match_pro[0]
                    match_anti = match_anti[0]
                    if(match_pro=="he" or match_pro=="He" or match_pro=="Men" or match_pro=="men" or match_pro=="man" or match_pro=="son" or match_pro=="Boys" or match_pro=="boys" or match_pro=="Boy" or match_pro=="boy" or match_pro=="Dad" or match_pro=="dad" or match_pro=="His" or match_pro=="his" or match_pro=="male" or match_pro=="uncle"):
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
    elif(dataset_name=="RedditBias"):
        df = pd.read_csv(dataset_path+'reddit_comments_gender_male_biased_test_reduced.csv')
        cleaned_pro_sentences = df['comments'].tolist()
        cleaned_anti_sentences = df['comments_processed'].tolist()
        female_targ = df['initial_demo'].tolist()
        male_targ = df['replaced_demo'].tolist()

        dyn_emb_target_pairs = {} #Dynamic Embeddings of Target Pairs
        targ_choices = ["woman", "Women", "Woman", "women", "daughter", "son", "female", "Girls", "girls", "Boys", "boys", "girl", "Girl", "male", "uncle", "aunt", "man", "she", "She", "men", "Men", "His", "his", "Boy", "boy", "mom", "Mom", "Dad", "dad", "her", "Her", "He", "he"]
        targ_choices.sort(key=str.lower, reverse=True)
        #print(targ_choices)
        j = 0

        for i in range(0, len(cleaned_pro_sentences)):
            pro_sent = cleaned_pro_sentences[i] #Female
            anti_sent = cleaned_anti_sentences[i] #Male

            match_fem = ast.literal_eval(female_targ[i])[0]
            #print(match_fem)
            match_male = ast.literal_eval(male_targ[i])[0]
            #print(match_male)

            pro_dyn = get_targ_dyn(pro_sent, match_fem, model_name, tokenizer, model)
            anti_dyn = get_targ_dyn(anti_sent, match_male, model_name, tokenizer, model)

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
        norm_check = np.sum(np.abs(comp_vec)**2, axis=1)  # Should be 1 for each embedding
        #print(norm_check)

        outer_prod = np.outer(comp_vec, np.conj(comp_vec))

        trait_complex_emb.append(outer_prod)
        trait_names.append(cur_trait)
    
    targ_emb = dyn_emb_target_pairs["Male-Female"]
    dist_scores = []

    for male_emb, female_emb in targ_emb:
        b = 0
        temp_male = []
        temp_female = []

        while(b<10):
            #Iterating over 10 instances of traits (powerful, powerless, ....)
            cur_dim_vec = [] #Current set of Complex vectors of Traits (size 30 (positive, negative, ..))

            for a in range(b, len(trait_complex_emb), 10):
                cur_dim_vec.append(trait_complex_emb[a])

            b += 1

            male_prob = []
            female_prob = []

            for i in range(0, len(cur_dim_vec)):
                P_m = cur_dim_vec[i] #np.outer(cur_dim_vec[i], np.conj(cur_dim_vec[i]))

                temp_intf_m = np.dot(P_m, male_emb)
                intf_m = np.vdot(male_emb, temp_intf_m)

                temp_intf_f = np.dot(P_m, female_emb)
                intf_f = np.vdot(female_emb, temp_intf_f)

                male_prob.append(abs(intf_m))
                female_prob.append(abs(intf_f))

            temp_male.append(male_prob)
            temp_female.append(female_prob)

        male_emb_fin = np.mean(temp_male, axis=0)
        fem_emb_fin = np.mean(temp_female, axis=0)

        dist = wasserstein_distance(male_emb_fin, fem_emb_fin)
        dist_scores.append(dist)
    
    avg_dist = np.mean(dist_scores)
    #print(f"Wasserstein distance: {avg_dist}")
    #print(np.sum(dist_scores)/len(dist_scores))

    print(f"SSS Wasserstein distance for {model_name} is: {str(avg_dist)}")

if __name__ == "__main__":
    models = ["GPT-2", "GPT-2XL", "RoBERTa", "ALBERT", "BERT-base", "BERT-large"]
    dataset_name = ["Winobias", "Crows-pairs", "RedditBias"]
    get_SSS_wasserstein_distance(models[0], "WinoBias_TestData/", dataset_name[0]) 