
import os
import copy
import json
import re

from numpy.lib.arraysetops import isin

from .library.bert.modeling import PreTrainedBertModel
from .library.bert.modeling import BertConfig
from .library.bert.modeling import BertSequenceModel
from .library.bert.modeling import classifier_fct
from .library.bert.modeling import softmax_fct
from .library.bert.modeling import init_weights
from .library.bert.MLMDataPrep import getVocab
from .library.preprocessing import clean_raw_notes

import numpy as np
import pandas as pd

import torch
from torch import nn

from transformers import AutoTokenizer
# from nltk.tokenize import sent_tokenize

from typing import List
from typing import Union
NoneType = type(None)
from collections import namedtuple

import logging
from .library.logWrapper import getLogger

loggerLevel = logging.WARNING

logger = getLogger(__name__, 
                    level=loggerLevel,
                    consoleLevel=loggerLevel, 
                    tofile=False) # do not put to file
class Data():
    def __init__(self, **kwargs):
        for key,value in kwargs.items():
            setattr(self, key, value)


class ClinicianNote(Data):
    """A ClinicianNote object can be:
        * a lowest level output prediction result,
            like: mseLab, mseLabNeg, topicLab etc
        * a mid level classifier output,
            like: SentenceClassifier, MseClassifier etc
        * or the higherst level result that contains all the information
        
        
    Properties:
        self.classifierType         (str):      "MasterClassifier", "SentenceClassifier", "mseLab" etc.
        self.subLevelNotes         (dict):      dictionary to keep track of lower level ClinicalNotes,
                                                "MasterClassifier" will have "SentenceClassifier", "mseClassifier", ...
                                                "SentenceClassifier" will have "mseLab", "mseNeg", ...
        self.prob           (np.ndarray):       raw probability output from the model, keep a copy and not to be modified
        self.filteredProb   (np.ndarray):       identical to self.prob most of the time, unless the main 
                                                classifier veto the classification of the row as False
                                                
                                                self.filteredProb can be adjusted by the following methods/properties:
                                                    self.applyFilter
                                                    self.resetFilter
                                                    self.thresholds
                                                    self.filterSubLevel
        self.thresholds     (float, 
                            np.ndarray,
                            dict):              the thresholds suppresses probability that are too low to zero
        self.sparseProb     (np.ndarray):       produced from self.filteredProb and self.thresholds
        self.labelKeys
    """
    
    def __init__(self,
                 classifierType:str = None,
                 prob: np.ndarray = None,
                 labelKeys:Union[str, np.ndarray] = None,
                 thresholds: Union[float, np.ndarray] = 0.1,
                 mainOutput:str = None,
                 **kwargs):
        
        super(ClinicianNote, self).__init__(**kwargs)
        
        self.classifierType = classifierType
        self.mainOutput = mainOutput
        self.subLevelNotes = dict()
        
        if self.classifierType == self.mainOutput:
            self.contextClassifiers = dict()
        
        if classifierType == 'topicLab':
            # only topicLab will/should populate this
            self.fellowClassifiers = dict()
            # assigned after construction
            # self.maptoClassifer = kwargs['topicLabMaptoClassifer']
        
        if isinstance(prob, np.ndarray):
            # is an array is given, this must be the lowest level classifers
            
            # similar to self.filteredProb
            # only accessible from the lowest levels
            self.filters = dict()
            
            self.prob = prob # hidden and not accessible to users
            
            # if the result of the lowet level, then label keys are mandatory
            # refer the labelKeys.setter
            self.labelKeys = labelKeys
            
            # default thresholds
            # refer to thresholds.setter
            self.thresholds = thresholds
            
    def appendSubLevelClassifer(self, noteObj):
        self.subLevelNotes[noteObj.classifierType] = noteObj
        setattr(self, noteObj.classifierType, noteObj)
        
    def linkSentClasToOtherClassifer(self, noteObj):
        """Similar idea to self.linkMainClassifierToContextClassifer
        """
        if self.classifierType == "MasterClassifier" and \
            noteObj.classifierType != "SentenceClassifier":
                # allow the SentenceClassifier to gain access to other classifers 
                sentenceClassifier = getattr(self, 'SentenceClassifier')
                topicLab = getattr(sentenceClassifier, sentenceClassifier.mainOutput)
                topicLab.fellowClassifiers[noteObj.classifierType] = noteObj
    
    def linkMainClassifierToContextClassifer(self, noteObj):
        """This method is to be called by second level classifers, such as as SentenceClassifer or 
        MseClassifer; suppose if it is MseClassifer
        
        MseClassifer will have an attribute stored call "mainOutput", for which will be "mseLab"
        mseLab must already have be appended to MseClassifer as a SubLevelClassifer.
        
        Then this method will look for "mseLab" under MseClassifer's SubLevelClassifer, 
        retrive the object, then append a contextClassifer such as "mseC_neg" under "mseLab"
        This is for the future whereby the result from mseLab is needed to filter mseC_neg result
        So mseLab must be able to access mseC_neg from its own attributes
        """
        if self.mainOutput == noteObj.classifierType:
            return
        getattr(self, self.mainOutput).contextClassifiers[noteObj.classifierType] = noteObj
            
    @property
    def prob(self) -> np.ndarray:
        """
        prob getter
        
        if this is the lowest level of result, 
        then simply returns the array
        
        otherwise, the getter are to take the prob array 
        of the lower level and concatenate them together

        Returns:
            np.ndarray: [description]
        """
        if not self.subLevelNotes:
            return self.__prob
        else:
            # concatenate all the result from the subLevels as a single vector
            subLevelProbs = tuple(note.prob for note in self.subLevelNotes.values())
            subLevelProbs = np.concatenate(subLevelProbs, axis = 1)
            self.shape = subLevelProbs.shape
            return subLevelProbs
        
    @prob.setter
    def prob(self, probArray:np.ndarray):
        """Only the lowest level model output should call this setter method 
        """
        assert not self.subLevelNotes, f'Not allow to set, implementation in progress'
        assert '_ClinicianNote__prob' not in self.__dict__, f'Not allow to modify prob'
        
        self.__prob = probArray
        self.shape = self.__prob.shape
        
    @property
    def labelKeys(self):
        if not self.subLevelNotes:
            return self.__labelKeys
        
        labels = dict()
        for noteName, noteObj in self.subLevelNotes.items():
            labels[noteName] = noteObj.labelKeys
        return labels
    
    @labelKeys.setter
    def labelKeys(self, labelKeysDF: Union[str, 
                                           pd.DataFrame,
                                           pd.Series,
                                           list,
                                           tuple,
                                           dict]):
        """
        labelKeys is used to parse the sparseProb into readable format
        
        If a string object is provided, it will be assumed to be a:
        * directory to .json or .csv
        * "Negation"
        * "Historical"
        
        If given imput is a pd.DataFram, pd.Series, list, tuple
        Then the order of the values in the labelKeys are important, 
        the position of the input will coresponds to the values in the output of the model
        
        If given object is a dictionary, then the keys will be used as the mapping to readable values.
        However, the keys must be in string from zero to N, where N is the dimension of the output vector
        """
        assert not self.subLevelNotes, 'Not allow to set, implementation in progress'
        
        if isinstance(labelKeysDF, str):
            if labelKeysDF.lower().endswith('.csv'):
                # if it is a str, it might be a directories to a csv file
                labelKeysDF = pd.read_csv(labelKeysDF, dtype=str)
            elif labelKeysDF.lower().endswith('.json'):
                labelKeysDF = json.load(open(labelKeysDF)) # json dictionary
                
                if isinstance(labelKeysDF, str):
                    # if the json file contains a single str
                    # means that this text is the binary context label
                    # such as negation
                    self.labelKeys = labelKeysDF
                    return
                    
                # check if the keys of the dictionary is a sequence from 0 to N
                keys = tuple(int(i) for i in labelKeysDF.keys())
                keys = sorted(keys)
                assert keys == list(range(len(keys))), f'Key values must be from 0 to {len(keys)-1}'
                labelKeysDF = list((int(i), value) for i, value in labelKeysDF.items())
                # arrange it from zero to N
                labelKeysDF.sort(key = lambda x: x[0])
                labelKeysDF = (label for pos, label in labelKeysDF)
                labelKeysDF = pd.DataFrame(labelKeysDF, dtype=str)
            elif not re.findall('\W', labelKeysDF):
                labelKeysDF = labelKeysDF
            else:
                fileName = os.path.basename(labelKeysDF)
                raise Exception(f"{fileName} is not supported as labelKeys.\n"
                                "To use a binary labelKeys, please use only word char [A-Za-z]")
        elif isinstance(labelKeysDF, (pd.Series, list, tuple)):
            labelKeysDF = pd.DataFrame(labelKeysDF, dtype=str)
        
        if isinstance(labelKeysDF, pd.DataFrame):
            validLabelKeysDF = (len(labelKeysDF) == self.shape[1]) or (len(labelKeysDF) == self.shape[2])
            assert validLabelKeysDF, str('Wrong number of keys,'
                                        f' Error with {self.classifierType},'
                                        f' {self.shape} keys expected,'
                                        f' {len(labelKeysDF)} is given')
        elif isinstance(labelKeysDF, str):
            validLabelKeysDF = not re.findall('\W', labelKeysDF)
            assert validLabelKeysDF, str('Wrong BinaryKey format'
                                        f' Error with {self.classifierType},'
                                        f' string using only [A-Za-z] is expected,'
                                        f' {labelKeysDF} is given')
        self.__labelKeys = labelKeysDF
        
    @property
    def thresholds(self) -> Union[float, np.ndarray, dict]:
        """Threshold values are meant to be used as a view for the user only
        """
        if not self.subLevelNotes:
            # if there is no subLevel
            return self.__thresholds
        
        # else, get the thresholds from all 
        # subLevels
        modelThresholds = dict()
        
        for noteName, noteObj in self.subLevelNotes.items():
            modelThresholds[noteName] = noteObj.thresholds
            
        return modelThresholds
    
    @thresholds.setter
    def thresholds(self, thresh:Union[float, np.ndarray, dict]):
        """User are not supposed to use this threshold setter
        It will lead to some breaks in the code
        
        The user can set the threshold by using "applyThresholds" method
        """
        
        if not self.subLevelNotes:
            # there is no subLevel
            # finish all the checks
            # the lowest level Notes only accepts float or np.ndarray as the input
            if isinstance(thresh, float) and (thresh > 1 or thresh < 0):
                raise Exception('Float threshold must be within [0, 1]')
            elif isinstance(thresh, float):
                pass
            elif isinstance(thresh, np.ndarray) and len(thresh.shape)>1:
                raise Exception('Threshold should be one dimensional array')
            elif isinstance(thresh, np.ndarray) and len(thresh) != self.shape[1]:
                raise Exception(f'array Threshold expect a len of {self.shape[1]} but {len(thresh)} is given')
            elif isinstance(thresh, np.ndarray):
                pass
            elif isinstance(thresh, str):
                # assume as json directory
                thresh = json.load(open(thresh))
                self.thresholds = thresh
                return
            elif isinstance(thresh, dict):
                # check if the keys of the dictionary is a sequence from 0 to N
                keys = tuple(int(i) for i in thresh.keys())
                keys = sorted(keys)
                assert keys == list(range(len(keys))), f'Key values must be from 0 to {len(keys)-1}'
                thresh = list((int(i), value) for i, value in thresh.items())
                # arrange it from zero to N
                thresh.sort(key = lambda x: x[0])
                thresh = np.array(tuple(label for pos, label in thresh))
                self.thresholds = thresh
            else:
                raise TypeError(f'{type(thresh)} not accepted as thresholds')
            
            # apply the threshold on self.filteredProb to produce self.sparseProb
            self.__thresholds = thresh
        
        else:
            if isinstance(thresh, float):
                for noteName, noteObj in self.subLevelNotes.items():
                    # set a single threshold for all subLevel
                    noteObj.thresholds = thresh
            if isinstance(thresh, dict):
                # use the relevant thresholds in the dict to apply
                for noteName, threshValue in thresh.items():
                    noteObj = self.subLevelNotes.get(noteName, None)
                    if noteObj:
                        noteObj.thresholds = threshValue
                    else:
                        logger.warning(f'thresholds not applied to {noteName},\n'
                                            f'{noteName} is not found in {self.classifierType=}:\n'
                                            f'{self.subLevelNotes.keys()}.')
                        pass
                        
            elif isinstance(thresh, np.ndarray):
                # this is tricky 
                # is an array is given, this array must be usable by all the sub models, 
                # which is quite unlikely, hence unless there is only one sub models
                # this will be rejected and prompted with multiple thresholds
                assert len(self.subLevelNotes) == 1, str('Threshold is ambiguous,'
                                                         ' Use a dictionary to indicate the specific target')
                self.subLevelNotes.values()[0].thresholds = thresh

        # refresh the filters everytime there's a change in the thresholds
        if self.subLevelNotes or self.classifierType == self.mainOutput:
            # the context classifers has no rights to filterSubLevel
            self.filterSubLevel = self.filterSubLevel
                
    @property
    def filterSubLevel(self):
        if not self.subLevelNotes:
            try:
                # if there is no subLevel
                return self.__filterSubLevel
            except AttributeError:
                return True
        
        modelFilters = dict()
        for noteName, noteObj in self.subLevelNotes.items():
            if noteObj.classifierType == noteObj.mainOutput:
                modelFilters[noteName] = noteObj.filterSubLevel
        return modelFilters
    
    @filterSubLevel.setter
    def filterSubLevel(self, status:Union[bool, dict]):
        """
        This method can only be called by SecondLevel Classifers
        Every SentenceClassifier will have an attribute "fellowClassifiers"
        That will give access to other models that are linked to these
        SentenceClassifier result
        
        This method will go through the output from SentenceClassifier,
        if the mseLab of a text is 0, then SentenceClassifier will 
        go to the result of mseLab and empty the result from that output.
        
        When status is True, this property will 
        update all the filters from this level;
        When status is False, this property will
        remove the filters from this level.
        """
        if isinstance(status, dict):
            for classifier, classiferStatus in status.items():
                self.subLevelNotes[classifier].filterSubLevel = classiferStatus
            return 
        elif self.subLevelNotes:
            # if the input is a boolean and the receiver is a 
            # SentenceClassifer/MseClassifer/etc
            self.subLevelNotes[self.mainOutput].filterSubLevel = status
        elif self.classifierType == "topicLab":
            # basically only applicable to "topicLab" at this moment
            # topicLab does not support contextClassifiers for now
            self.__filterSubLevel = status
            if status:
                sentenceClassifierResult = self.sparseProb
                # this part is vulnerable to code break if 
                # some order in the config is not properly preserved
                # need to improve on this part of the code in the future
                for noteName, noteObj in self.fellowClassifiers.items():
                    i = self.maptoClassifer[noteName]
                    # for each TopicWiseModel
                    modelFilter = sentenceClassifierResult[:,i] <= 0
                    noteObj.applyFilter(self.classifierType, modelFilter)
            else:
                for i, (noteName, noteObj) in enumerate(self.fellowClassifiers.items()):
                    # remove the filter from this object
                    noteObj.resetFilter(self.classifierType)
        elif self.classifierType == self.mainOutput:
            # this branch is for mseLab, stressorLab etc.
            # do not accept mseNeg, stressorNeg etc
            # the input is a boolean and the receiver is a 
            # mseLab/stressorLab/etc
            self.__filterSubLevel = status
            if status:
                mainClassiferOutput = self.sparseProb
                modelFilter = mainClassiferOutput <= 0
                for i, (noteName, noteObj) in enumerate(self.contextClassifiers.items()):
                    noteObj.applyFilter(self.classifierType, modelFilter)
            else:
                for i, (noteName, noteObj) in enumerate(self.contextClassifiers.items()):
                    noteObj.resetFilter(self.classifierType)
                
    @property
    def filteredProb(self) -> np.ndarray:
        """
        In most of the situations, filteredProb is the same as self.prob
        unless if SentClassifier is involved, and SentClassifier dictates 
        that a particular row should not have any result, then the result from 
        that particular row will be filtered away
        
        Or suppose a mseLab classifier dictates that a particular row does not have 
        a particular label; but mseC_neg predicted a negation in that label; then 
        a filter from mseLab will be used to set that value from mseC_neg to zero
        
        technically, filteredProb should only be called when self.sparseProb is 
        called at the lowest level, thus only the lowest level of classifiers 
        should have a filteredProb

        Raises:
            AssertionError: Prevent higher level classifier from accessing this property

        Returns:
            np.ndarray
        """

        assert not self.subLevelNotes, 'Error, filteredProb not accessible to high level ClinicalNote'
        
        if self.classifierType == 'topicLab':
            # topicLab has the highest priority
            # it has no filters at all
            # because no other classifier will veto the 
            # result of SentenceClasifier
            return self.prob.copy()
        
        modelProb = self.prob.copy()
        for filterArray in self.filters.values():
            # rows that SentenceClassifier does not agree 
            # with be supressed to 0
            if filterArray.shape == modelProb.shape:
                # element-wise filter
                modelProb[filterArray] = 0
            elif filterArray.shape == modelProb.shape[1:]:
                # row wise filter
                modelProb[:,filterArray] = 0
            else:
                raise Exception('Wrong filter dimension')
        return modelProb
            
    def applyFilter(self, filterSource:str, filterObj:np.ndarray) -> None:
        """[summary]

        Args:
            filterSource (str): a string indicating the source of the filter
                                if a mseLab apply a filter on mseC_neg, then
                                the filterSource = 'mseLab'
            filterObj (np.ndarray): [description]

        Raises:
            TypeError: [description]
        """
        if not self.subLevelNotes:
            if isinstance(filterObj, str):
                filterObj = json.load(open(filterObj))
                self.applyFilter(filterSource = filterSource, 
                                 filterObj = filterObj)
                return
            elif isinstance(filterObj, dict):
                # check if the keys of the dictionary is a sequence from 0 to N
                keys = tuple(int(i) for i in filterObj.keys())
                keys = sorted(keys)
                assert keys == list(range(len(keys))), f'Key values must be from 0 to {len(keys)-1}'
                filterObj = list((int(i), value) for i, value in filterObj.items())
                # arrange it from zero to N
                filterObj.sort(key = lambda x: x[0])
                filterObj = np.array(tuple(label for pos, label in filterObj))
                self.applyFilter(filterSource=filterSource, 
                                 filterObj = filterObj)
                return
            elif isinstance(filterObj, np.ndarray):
                pass
            else:
                raise TypeError('Wrong type for filters\n'
                                f'{filterSource=}\n'
                                f'{type(filterSource)=}')
            
            self.filters[filterSource] = filterObj
        else:
            for noteName, noteObj in self.subLevelNotes.items():
                noteObj.applyFilter(filterSource, filterObj)
                
    def resetFilter(self, filterSource:str = None) -> None:
        """Indicate a filterSource to remove the filter from the list.
        Else, all filters will be removed if no source is indicated
        """
        if not self.subLevelNotes:
            if filterSource:
                if filterSource in self.filters:
                    del self.filters[filterSource]
            else:
                self.filters = dict()
        else:
            for noteName, noteObj in self.subLevelNotes.items():
                noteObj.resetFilter()
                
    @classmethod
    def binaryToReadable(cls, 
                         sparseProb: np.ndarray, 
                         keyMapper: Union[pd.DataFrame, str],
                         clean:bool = True) -> List[List]:
        """
        using this object, the result of the self.sparseProb will be converted to 
        human readable text

        Args:
            sparseProb (np.ndarray): [description]
            keyMapper (Union[pd.DataFrame, str]): [description]
            clean (bool, optional): [description]. Defaults to True.

        Returns:
            List[List]: [description]
        """        
        result = []
        for row in sparseProb:
            rowResult = []
            # self.logger.debug(row)
            for index, value in enumerate(row):
                # self.logger.debug(value)
                if isinstance(value, np.ndarray) and any(value):
                    # output is from a multi-dimension one-hot contextClassifier
                    # this part of the code is not yet tested for multi-dimension context
                    text = " - ".join(keyMapper.iloc[value>0,:])
                    rowResult.append(text)
                elif value > 0:
                    # map for the value
                    # self.logger.debug(str(index))
                    # self.logger.debug(f"{classifier} {modelOutput} {index}")
                    if isinstance(keyMapper, str):
                        # for binary contextClassifier's labelKeys
                        rowResult.append(keyMapper)
                    else:
                        # for mainOutput such as mseLab, StressorLab
                        text = " - ".join(keyMapper.iloc[index,:])
                        # self.logger.debug(text)
                        rowResult.append(text)
                else:
                    if not clean:
                        rowResult.append(None)
            if clean:
                rowResult = rowResult if rowResult else None
            result.append(rowResult)
        return result
    
    @property
    def sparseProb(self):
        """
        reliant on thresholds and filteredProb
        For each elements in the filteredProb, if the value of the element is
        lower than threshold; the value will be suppressed to zero
        producing a sparse matrix
        """
        if not self.subLevelNotes:
            sparseProb = self.filteredProb
            sparseProb[sparseProb < self.thresholds] = 0
            return sparseProb
        else:
            # concatenate all the result from the subLevels as a single vector
            subLevelProbs = tuple(note.sparseProb for note in self.subLevelNotes.values())
            subLevelProbs = np.concatenate(subLevelProbs, axis = 1)
            self.shape = subLevelProbs.shape
            return subLevelProbs
    
    @property
    def readable(self) -> List[List]:
        """
        Retrieving data from self.sparseProb and labelKeys;
        using the two to produce readable version of the outcome

        Returns:
            List[List]: [description]
        """
        if not self.subLevelNotes and self.classifierType != self.mainOutput:
            # for context classifiers
            return self.binaryToReadable(self.sparseProb, self.labelKeys, clean = False)
        elif self.classifierType == self.mainOutput and not self.contextClassifiers:
            # for mseLab or stressorLab but that has no context classifier given
            return self.binaryToReadable(self.sparseProb, self.labelKeys, clean = True)
        elif self.classifierType == self.mainOutput:
            # for mseLab or stressorLab that already HAS a context classifier
            labNcontexts = [self.binaryToReadable(self.sparseProb, self.labelKeys, clean = False),]
            # append the result from each context classifier
            for contextClassifier in self.contextClassifiers.values():
                labNcontexts.append(contextClassifier.binaryToReadable(contextClassifier.sparseProb, 
                                                                       contextClassifier.labelKeys, 
                                                                       clean = False))
            labNcontexts = list(zip(*labNcontexts))
            for index,v in enumerate(labNcontexts):
                # zip the lab with the contexts into a list
                v = set(zip(*v))
                v.discard((None, None)) # this part is tricky
                # you wont want too many None in the readable, 
                # otherwise it defeats the purpose of readable
                # this part of the code will not work properly if I have more than 1 
                # context classifier for each classifier
                v = v if v else None
                labNcontexts[index] = v
            return labNcontexts
        else:
            result = dict()
            for note in self.subLevelNotes.values():
                if not note.subLevelNotes and note.classifierType != note.mainOutput:
                    # if it is a contextClassifier
                    continue
                result[note.classifierType] = note.readable
            return result
    
    @property
    def summarySparse(self):
        """
        flatten the prediction outcome of all N sentences into 1;
        the output is an Union of all sentences; 
        """
        # for each column(aka each label), if there is any value true in that column, 
        # the resultant value of that column is True
        return self.sparseProb.any(axis=0).astype(int)
    
    @property
    def summaryReadable(self):
        """flatten the prediction outcome of all N sentences into 1;
        the output is an Union of all sentences
        """
        if not self.subLevelNotes:
            # return self.labelKeys.loc[self.summarySparse>0,:].apply(lambda x: " - ".join(x), axis = 1).tolist()
            result = set()
            for i in self.readable:
                if i:
                    result.update(i)
            # result.union(*self.readable) # wont work because there might be None present in readable
            return result
        else:
            result = dict()
            for note in self.subLevelNotes.values():
                if not note.subLevelNotes and note.classifierType != note.mainOutput:
                    # if it is a contextClassifier
                    continue
                result[note.classifierType] = note.summaryReadable
            return result
        
class BertForPackage(PreTrainedBertModel):
    
    def __init__(self, 
                 modelConfig: Union[str, dict], 
                 input_size:int = 768):
        """

        Args:
            modelConfig (Union[str, dict]): accept a str that is a path to a json file; 
                                            or a dictionary that is the json config 
            input_size (int, optional): maximum input len. Defaults to 768.
            loggerLevel (int, optional): Defaults to logging.WARNING.
        """        
        
        if isinstance(modelConfig, str):
            modelConfig = json.load(open(modelConfig))
        
        bertConfig = BertConfig(os.path.join(modelConfig['BertPath'],
                                             'bert_config.json'))
        
        super(BertForPackage, self).__init__(bertConfig)
        self.modelConfig = modelConfig
        
        self.vocab = getVocab(os.path.join(modelConfig['BertPath'],
                                           'vocab.txt'),
                              getdict=False)
        self.modelConfig['Tokenizer'] = json.load(open(os.path.join(self.modelConfig['BertPath'],
                                                                    'tokenizer_config.json')))
        nizer = AutoTokenizer.from_pretrained(self.modelConfig['Tokenizer']['tokenizer'])
        model = BertSequenceModel.from_pretrained(self.modelConfig['BertPath'])
        
        self.tokenizer = nizer
        self.bert = model
        
        
        preprocess = clean_raw_notes.preprocess(modelConfig['preprocessing'])
        self.sent_tokenize = preprocess.run
        
        self.SupportedModel = tuple(model for model, status in self.modelConfig['SupportedModel'].items() if status)
        
        if self.modelConfig["Device"].lower() == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.modelConfig["Device"].lower())
        logger.info(f"Model in: {self.device}")
        
        self.bert.to(self.device)
        
        for classifierName, on in self.modelConfig['SupportedModel'].items():
            if not on:
                continue
            
            classifierConfig = json.load(open(os.path.join(self.modelConfig[classifierName]['path'],
                                                           "modelConfig.json")))
            self.modelConfig[classifierName]['config'] = classifierConfig
            self.modelConfig[classifierName]['mainOutput'] = classifierConfig[classifierName]['modelOutput'][0]
            self.modelConfig[classifierName].update(
                dict(labelKeys = dict(), thresholds = dict(), filters = dict())
            )
            
            # adding the lebelKeys, thresholds and filters to the config
            for dataDocBase in os.listdir(self.modelConfig[classifierName]['path']):
                dataDocSplit = list(dataDocBase.split("_", maxsplit=2))
                if dataDocSplit[0] != 'data':
                    continue
                dataDocSplit[2] = dataDocSplit[2].replace('.json', '')
                self.modelConfig[classifierName][dataDocSplit[1]][dataDocSplit[2]] = os.path.join(self.modelConfig[classifierName]['path'],
                                                                                 dataDocBase)
            
            if classifierConfig[classifierName]['trainingParams']['output_token_level']: 
                # AggregationNN: (batch, seq_length, 768) --> (batch, 768)
                # 1. AttentionNN 
                for outputLayer in classifierConfig[classifierName]['modelOutput']:
                    outputLayerAttNN = outputLayer + "_attNN"
                    params = classifierConfig[classifierName]['modelArchitecture'][outputLayerAttNN]
                    setattr(self, outputLayerAttNN, classifier_fct(clf_input_size=input_size,
                                                              clf_hidden_size=params['hidden_size'],
                                                              clf_output_size=1,
                                                              clf_activation=params['activation'],
                                                              clf_output_activation="Sigmoid",
                                                              clf_dropout_prob=params['dropout_prob']
                                                              ).apply(init_weights)
                            )
                    getattr(self, outputLayerAttNN).to(self.device)
            
            for outputLayer in classifierConfig[classifierName]['modelOutput']:
                params = classifierConfig[classifierName]['modelArchitecture'][outputLayer]
                setattr(self, outputLayer, classifier_fct(clf_input_size=input_size,
                                                            clf_hidden_size=params['hidden_size'],
                                                            clf_output_size=params['n_labels'],
                                                            clf_activation=params['activation'],
                                                            clf_output_activation=params['output_activation'],
                                                            clf_dropout_prob=params['dropout_prob']
                                                            ).apply(init_weights)
                        )
                getattr(self, outputLayer).to(self.device)
        
            # import and load the parameters
            pretrained_dict = torch.load(os.path.join(self.modelConfig[classifierName]['path'], 
                                                      "pytorch_model.bin"),
                                         map_location=self.device) # model is stored in the device specified
            model_dict = self.state_dict()
            pretrained_dict_load = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            
            extraParams = {k: v for k, v in pretrained_dict.items() if k not in model_dict}
            if extraParams:
                logger.warning(f"Extra parameters found!")
            model_dict.update(pretrained_dict_load)
            self.load_state_dict(model_dict)
            self.eval() # turn off all gradient and dropOuts etc
            
                
    def tokenize(self, 
                 inputSent: List[str]) -> Data:
        """[summary]
        Tokenise each string in the list, make use of the config options specified in 
        BertPath, under tokenizer_config.json 
        to control the len of output, truncation and max_length

        Args:
            inputSent (List[str]): A list of string, where each string is a sentence

        Raises:
            TypeError: If the input is not a list of string, 
                       or cannot be converted into a list

        Returns:
            Data
        """        
             
        try:
            inputSent = list(inputSent)
        except Exception as e:
            raise TypeError('You need to provide a list of strings(sentences)')
        
        tokenizerOutput = self.tokenizer(inputSent,
                                         padding=self.modelConfig['Tokenizer']['padding'],
                                         truncation=self.modelConfig['Tokenizer']['truncation'],
                                         max_length=self.modelConfig['Tokenizer']['max_length'],
                                         return_tensors='pt')
        
        input_ids = tokenizerOutput['input_ids']
        input_tokens = [[self.vocab[i] for i in sentence] for sentence in input_ids]
        # put to device
        input_ids = tokenizerOutput['input_ids'].to(self.device)
        token_type_ids = tokenizerOutput['token_type_ids'].to(self.device)
        attention_mask = tokenizerOutput['attention_mask'].to(self.device)
        
        # self.logger.debug(f'data device: {input_ids.get_device()}')
        encoded_layers, pooled_output = self.bert(input_ids,
                                                  token_type_ids,
                                                  attention_mask, 
                                                  output_all_encoded_layers=False)
        
        tokenLevel_layer = encoded_layers
        attention_mask = attention_mask.unsqueeze(dim=2) # used in token level models
        # encoded_layers.shape() -> list of (sentences=N, token_len=64, embedding_sim=768) -> len = 12 or 1 
        # pooled_output.shape() -> (sentences=N, embedding_sim=768)
        
        tokenizerOutput = Data(
            inputSent = inputSent,
            input_tokens = input_tokens,
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
            pooled_layer = pooled_output,
            tokenLevel_layer = tokenLevel_layer
        )
        return tokenizerOutput
                    
                    
    def classify(self, 
                 classifier:str,
                 text:Union[str, List[str], Data]) -> ClinicianNote:
        """
        Allow to specify to use which classifier only
        
        Args:
            classifier (str): classifier name to use, acceptable inputs 
                              are from self.SupportedModel
            text (Union[str, List[str], Data]): User input for analysis, 
                                                can be provided in string or list of string,
                                                just like processDocument.
                                                
                                                Accepts a third type of input, that is a Data
                                                instance, not recommended for user. 

        Returns:
            ClinicianNote
        """        
        
        assert classifier in self.SupportedModel, str(f"{classifier} not supported, "
                                                      "please choose one of the following:"
                                                      f"\n{self.SupportedModel}")
        
        if isinstance(text, str):
            # given a text, means that this is a document level prediction
            text = self.sent_tokenize(text)
        if isinstance(text, (list, tuple, pd.Series, np.ndarray)):
            tokeniseOutput = self.tokenize(text) # Data instance
        if isinstance(text, Data):
            tokeniseOutput = text
            
        tokeniseOutputCache = tokeniseOutput if self.modelConfig['KeepTokens'] else None
        
        pooled_output = tokeniseOutput.pooled_layer
        
        classifierOutput = ClinicianNote(classifierType=classifier,
                                         tokeniseOutput=tokeniseOutputCache,
                                         mainOutput = self.modelConfig[classifier]['mainOutput'])
        
        classifierConfig = self.modelConfig[classifier]['config']
        # each classifier may have multiple layers of predictions
        for outputLayer in classifierConfig[classifier]['modelOutput']:
            
            if classifierConfig[classifier]['trainingParams']['output_token_level']: 
                # AggregationNN: (batch, seq_length, 768) --> (batch, 768)
                # 1. AttentionNN
                outputLayerAttNN = outputLayer + "_attNN"
                
                logger.debug(f'tokeniseOutput.tokenLevel_layer is cuda: {tokeniseOutput.tokenLevel_layer.is_cuda}')
                attNN_output = getattr(self, outputLayerAttNN)(tokeniseOutput.tokenLevel_layer) # attNN_output.size()=[128, 64, 1]
                attNN_output = softmax_fct(attNN_output, tokeniseOutput.attention_mask) # attNN_output.size()=[128, 64, 1]

                # 2. Apply weights
                pooled_output = tokeniseOutput.tokenLevel_layer * attNN_output # pooled_output3.size()=[128, 64, 768]

                pooled_output = torch.sum(pooled_output, dim=1) # pooled_output3.size()=[128, 768]
                
            output = getattr(self, outputLayer)(pooled_output).detach().cpu().numpy()
        
            logger.debug(f'{outputLayer}, shape  = {output.shape}')
            output = ClinicianNote(classifierType = outputLayer,
                                   prob = output,
                                   labelKeys = self.modelConfig[classifier]['labelKeys'][outputLayer],
                                   thresholds = self.modelConfig[classifier]['thresholds'][outputLayer],
                                   tokeniseOutput=tokeniseOutputCache,
                                   mainOutput = self.modelConfig[classifier]['mainOutput'])
            # output.filterSubLevel = output.classifierType == output.mainOutput
            # output.thresholds = self.modelConfig[classifier]['labelThresholds'][outputLayer]
            if outputLayer == 'topicLab':
                output.maptoClassifer = json.load(open(self.modelConfig['topicLabMaptoClassifer']))
            
            if outputLayer in self.modelConfig[classifier]['filters']:
                filteObj = self.modelConfig[classifier]['filters'][outputLayer]
                # a special filter for each particular model
                # for example so negation model may specify some value to be always false
                output.applyFilter(outputLayer, filteObj)

            classifierOutput.appendSubLevelClassifer(output)
            classifierOutput.linkMainClassifierToContextClassifer(output)
            
        
        # apply the filter on the fellow classifiers if needed
        classifierOutput.filterSubLevel = self.modelConfig[classifier]['filterSubLevel']
        classifierOutput.thresholds = classifierOutput.thresholds
        return classifierOutput

    def forward(self, 
                inputSent: List[str]) -> ClinicianNote:
        """
        Given a list of sentences, the forward pass will tokenised the
        sentences, passed into all "Supported Classifiers"
        
        The results will be collated and returned in an ClinicianNote instance

        Args:
            inputSent (List[str]): sentences to be classifed, 
                                    a list of str, or an iterable or strings that can
                                    be converted to a list of strings

        Raises:
            e: If your provided input cannot be converted into a list of strings

        Returns:
            ClinicianNote: the returned result containing:
                                            inputSent,
                                            input_tokens,
                                            input_ids,
                                            token_type_ids,
                                            attention_mask,
                                            <<ModelOutput objects>>
        """
        
        tokeniseOutput = self.tokenize(inputSent)
        tokeniseOutputCache = tokeniseOutput if self.modelConfig['KeepTokens'] else None
        
        MasterOutput = ClinicianNote(classifierType='MasterClassifier',
                                     # MasterClassifier is the highest level 
                                     # containing SentenceClassifer, MseClassifier etc
                                     mainOutput = 'SentenceClassifier',
                                     tokeniseOutput=tokeniseOutputCache)

        for classifier, v in self.modelConfig['SupportedModel'].items():
            if not v:
                continue
            
            classifierOutput = self.classify(classifier, text = tokeniseOutput)
            
            if 'SentenceClassifier' in self.SupportedModel:
                # allow SentenceClassifier to gain access to filter other classifiers
                MasterOutput.linkSentClasToOtherClassifer(classifierOutput)
            MasterOutput.appendSubLevelClassifer(classifierOutput)
        
        
        if self.modelConfig['KeepTokens']:
            # toggle KeepTokens to state if the result should keep the copy 
            # of input string and the tokens produced, if set to True, 
            # beware that ram consumption will be higher
            for k,v in tokeniseOutput.__dict__.items():
                if isinstance(v, torch.Tensor):
                    # remove the input tensors from GPU
                    # otherwise this copy accumulatively occupy more space on GPU Ram
                    # copying the values to CPU ram might be time consuming
                    # ram consumption will be high too
                    v = v.detach().cpu().numpy()
                    setattr(tokeniseOutput, k, v)

        return MasterOutput
    
    def processDocument(self, text:Union[str, List[str]]) -> ClinicianNote:
        '''
        process the given text and return a ClinicianNote instance containing information
        about the analysis of the text
        
        Args:
            text:Union[str, List[str]]:         This methods accepts two type of input.

                                                If the given input is a string, the model will proceed with tokenization,
                                                breaks the given string into suitable sentences, then finally pass the 
                                                sentences to the models.
                                                
                                                If the given input is a list of string, then holbert will skip the 
                                                tokenizing process
        '''
        if not text:
            logger.warning('No input received.\n'
                                'Please provide a string or a list of string.')
            return None
        elif isinstance(text, str):
            text = self.sent_tokenize(text)
        elif isinstance(text, (list, tuple, pd.Series, np.ndarray)):
            text = text
        else:
            raise TypeError('Wrong format given for text analysis.\n'
                            'Please provide a string or a list of string.')
        
        forwardOutput = self.__call__(text)
        return forwardOutput

    
        
        
        
        
    
    
    
    
    
    