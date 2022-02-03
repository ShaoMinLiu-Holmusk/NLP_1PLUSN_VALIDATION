
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path


script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from typing import List
from numpy import ndarray
from pandas import Series

class FeatureInterface:
    def __init__(self, name:str, subModule:str,
                 config:dict, txtLocation:str) -> None:
        self.name = name
        self.subModule = subModule
        self.config = config
        self.txtLocation = txtLocation
        
    def extract(self, documents:Series) -> ndarray:
        """given a list of document name,
        extract the features of the document
        in an ndarray

        Parameters
        ----------
        documents : pd.Series
            A series of the name of the document

        Returns
        -------
        ndarray
            [description]
        """
        raise NotImplementedError

    '''
    def fit(self, document: List[str], y:ndarray)-> None:
        """Not all Feature will require training, 
        so some can just overwrite and pass

        Parameters
        ----------
        X : List[str]
            list of strings, each string a document
        y : ndarray
            representation of the the documents
        """
        raise NotImplementedError
    
    def predict(self, X:List[str])-> ndarray:
        """Take in list of documents, return the 
        matrix representation of the model

        Parameters
        ----------
        document : List[str]
            [description]

        Returns
        -------
        ndarray
            [description]
        """
        raise NotImplementedError
    '''