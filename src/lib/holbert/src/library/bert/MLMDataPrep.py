'''
This script is the translation of ipynb for the dataPrepMLM variants


'''

import pandas as pd
import numpy as np
import json

from transformers import BertTokenizerFast
from transformers import AutoTokenizer

import collections
import random
import re

import sys
import os
from tqdm.notebook import tqdm

from typing import Iterable
from typing import Tuple
from typing import List

def dataLoading(dirData:str,
                batches:int = 1) -> pd.DataFrame:
    """

    Args:
        dirModelConfig (str): [description]
        dirData (str): is the directory of the csv files to be read
                        csv files must end with 'Batch{number}.csv'
        batches (int): total number of batches there are, if there ia only one file
                        then there is only one batch

    Returns:
        pd.DataFrame: [description]
    """
    
    
    modelConfig = json.load(open(dirModelConfig))
    # vocabPath = os.path.join(modelConfig['Path']['bioBERT'], modelConfig['Path']['vocab'])
    
    assert os.path.exists(dirData), 'Data Directory not exist!'
    
    # read CSV

    df_batches  = [pd.read_csv(os.path.join(dir_data, f'Batch{i}.csv'), index_col=0) for i in range(1,batches+1)]
    for each_df in df_batches:
        print(each_df.columns.tolist())
        print(each_df.shape)

    df_full = pd.concat(df_batches)
    # df_full.reset_index(inplace = True)
    df_full.shape
    
    df_full = df_full.drop_duplicates('sentence')
    
    df_full = df_full.dropna(subset=['sentence'])
    df_full.reset_index(inplace = True, drop = True)
    
    return df_full

# create_masked_lm_predictions

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
def create_masked_lm_predictions(
    tokens: Iterable, 
    masked_lm_prob: float,
    max_predictions_per_seq: int, 
    vocab_words, rng) -> Tuple[list, list, list]:
    """[summary]

    Args:
        tokens (Iterable): contains strings of tokens
        masked_lm_prob (float): probability of masking
        max_predictions_per_seq (int): maximum number of smaking
        vocab_words ([type]): list of vocabulary
        rng ([type]): unknown

    Returns:
        Tuple[list, list, list]: [
            a list of tokens after masking,
            indices of all masked token,
            origin representation of the masked tokens,
        ]
    """

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        # enumerating using position in the sentence
        if (token == "[CLS]") or (token == "[SEP]") or \
            (token == "[") or (token == "]") or (token == "[") or \
            (token == ".") or (token == ",") or (token == "-"):
            # do not add to the candidate indexes if its one of these tokens
            continue
        cand_indexes.append(i)

    # randomise the order
    # inplace?
    rng.shuffle(cand_indexes)
    
    # `tokens` is not a list?
    output_tokens = list(tokens)

    # a number between 1 and max_predictions_per_seq
    num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break

        # why would an index be covered twice?
        if index in covered_indexes:
            continue

        # add it if it has not been covered
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
          # 10% of the time, keep original
          if rng.random() < 0.5:
            masked_token = tokens[index]
          # 10% of the time, replace with random word
          else:
            masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
        # replace the value with the masking
        output_tokens[index] = masked_token

        # keep track of masked instances
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # unpacking the values
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def clean_token(tok:str)-> List[str]:
    """Apply to rows of a dataframe, in the assumption that each 
    row is a str representation of List[str]

    This function may not be neccessary in this notebook

    Args:
        tok (List[str]): A list of tokens

    Returns:
        List[str]: A list of clean tokens
    """

    s = tok.split(', ')
    l=[]
    for i in range(len(s)):
        if (i==0):
            l.append(s[i][2:-1])
        elif (i==len(s)-1):
            l.append(s[i][1:-2]) 
        else:
            l.append(s[i][1:-1])
    return l


def tokenizing(sentence: str,
               max_length: int,
               tokenizer) -> List[str]:
    """[summary]

    Args:
        sentence (str): sentence

    Returns:
        List[str]: list of tokens
    """

    encodings = tokenizer(sentence,
                            return_offsets_mapping=True,
                            padding='max_length', 
                            truncation=True, 
                            max_length=max_length)

    return pd.Series([encodings.input_ids, encodings.attention_mask],
                     index = ['input_ids', 'attention_mask'])

# testing
# tokenizing("On [Date], her glucose was 70.")

def getVocab(dirVocab, getdict = True):
    with open(dirVocab, "r") as file:
        vocab_words = file.readlines()
        vocab_words = [re.sub('\n','',i) for i in vocab_words]
        # a dictionary of the word an its index
        if getdict:
            vocab_words_dict = dict(zip(vocab_words, range(len(vocab_words))))
        else:
            return vocab_words
    return vocab_words, vocab_words_dict

def contructMLMLabels(row, vocab_words_dict):
#     lab = []
    lab_str = []
    j=0
    
    t=row['mlm']
    l=row['tokens_mlm']
    positions=row['mlm_positions']
    
    # for each token in tokens_mlm
    for i, n in enumerate(l):
        # if this token is masked
        if i in positions:
            # lab.append(1)
            # find the masked word
            # find the vocab index of the masked word
            # add it to lab_str
            lab_str.append(vocab_words_dict[t[j]])
            j+=1
        else:
            # lab.append(0)
            lab_str.append(-1)
    return lab_str


def prepMLMBioBert(dirModelConfig:str,
                   dirData:str, batches:int,
                   rng, max_predictions_per_seq, masked_lm_prob,
                   tokenizerModel = AutoTokenizer):
    
    df_nodupl = dataLoading(dirData = dirData,
                            batches = batches)
    
    modelConfig = json.load(open(dirModelConfig))
    vocabPath = os.path.join(modelConfig['Path']['bioBERT'], modelConfig['Path']['vocab'])
    
    tokenizer = tokenizerModel.from_pretrained(modelConfig['TrainingParams']['tokenizer'])
    max_length = modelConfig['TrainingParams']['max_seq_len']
    
    # attention_mask is if there is a word
    df_nodupl[['input_ids','attention_mask']] = df_nodupl.apply(lambda row: tokenizing(row.sentence, 
                                                                                       max_length,
                                                                                       tokenizer), 
                                                                result_type='expand', axis=1)  
    
    df_nodupl['token_len'] = df_nodupl.attention_mask.apply(sum)
    
    
    vocab_words, vocab_words_dict = getVocab(dirVocab = vocabPath)
    df_nodupl['tokens_lm'] = df_nodupl['input_ids'].apply(lambda x: [vocab_words[i] for i in x])
    
    # initialising variables
    # rng = random.Random(12345)
    # max_predictions_per_seq=20
    # masked_lm_prob = 0.15
    
    def lm_input(row):
        tok =row['tokens_lm']

        output_tokens, masked_lm_positions,masked_lm_labels = create_masked_lm_predictions(
            tokens=tok, masked_lm_prob=masked_lm_prob,
            max_predictions_per_seq=max_predictions_per_seq, 
            vocab_words=vocab_words, rng=rng)

        lm_input = (output_tokens)
        labels = (masked_lm_labels)
        lm_positions = (masked_lm_positions)
        return [lm_input,labels,lm_positions]
    
    df_nodupl[['tokens_mlm','mlm', 'mlm_positions']] = df_nodupl.apply(lm_input, axis=1, result_type='expand')
    
    df_nodupl['mlmLab'] = df_nodupl.apply(contructMLMLabels, vocab_words_dict = vocab_words_dict, axis=1)
    
    df_nodupl['input_ids'] = df_nodupl['tokens_mlm'].apply(lambda x: [vocab_words_dict[i] for i in x])
    
    
    df_nodupl['length'] = df_nodupl['attention_mask'].apply(lambda x: np.array(x).sum())
    df_nodupl['length_cat'] = df_nodupl['length'].apply(lambda x: 2 if x>50 else 1)
    
    # only select greater than 10
    df_nodupl = df_nodupl[df_nodupl['length']>10].reset_index(drop=True)
    
    df_nodupl['sentencesNo'] = pd.Series(range(df_nodupl.shape[0]))
    
    return df_nodupl


def picklingFull(df, dirTarget, pickleName):
    dir_pickle = os.path.join(dirTarget, pickleName)
    df.to_pickle(dir_pickle)

def picklingChunk(df, chunksNums, dirTarget, pickleName):
    totalSize = df.shape[0]
    chunkSize = totalSize//10
    startingIndex = 0

    pbar = tqdm(total=totalSize)
    while startingIndex <= totalSize:
        endingIndex = startingIndex + chunkSize
        pickleName = pickleName + f'_chunkIndex{startingIndex}.pkl'
        df.iloc[startingIndex:endingIndex,:].to_pickle(os.path.join(dirTarget, pickleName))
        pbar.update(chunkSize)
        startingIndex = endingIndex
    pbar.close()
    

if __name__ == "__main__":
    # to generate bioBert Data
    data = prepMLMBioBert(dirModelConfig = '../config/modules/model/modelConfigMLM_ClinicalBioBERT.json',
                    dirData = '../data/intermediate/preprocessed/', 
                    batches = 6,
                    rng = random.Random(12345), 
                    max_predictions_per_seq = 20, 
                    masked_lm_prob = 0.15,
                    tokenizerModel = AutoTokenizer)
        
    picklingFull(data, 
                dirTarget = '../data/intermediate/MLM_bioBERT', 
                pickleName = 'curated_data_mlm_tokenized_allBatches_withTokenizer_greater_10_biobert_chunkFull.pkl')

    picklingChunk(data, chunksNums = 10, 
                dirTarget = '../data/intermediate/MLM_bioBERT',
                pickleName = 'curated_data_mlm_tokenized_allBatches_withTokenizer_greater_10_biobert')
        
    
    
    
    