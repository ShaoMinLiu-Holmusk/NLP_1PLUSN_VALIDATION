
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path


script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from typing import List
from numpy import ndarray, array

class ModelInterface:
    def __init__(self, config:dict) -> None:
        self.config = config
    
    def fit(self, X: ndarray, y:array)-> None:
        """

        Parameters
        ----------
        X : ndarray
            matrix representation of the document, 
            numerical element
        y : array
            boolean outcome
        """
        raise NotImplementedError
    
    def predict_proba(self, X:ndarray)-> array:
        """Take in list of documents, return the prediction in probability

        Parameters
        ----------
        document : ndarray
            matrix representation of the document

        Returns
        -------
        array
            logit output of the prediction, 
            make sure it is an 1-d array
            squeeze it for sure
        """
        raise NotImplementedError