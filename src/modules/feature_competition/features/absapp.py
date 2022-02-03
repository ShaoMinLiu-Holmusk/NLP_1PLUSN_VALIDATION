
from pathlib import PosixPath
from modules.feature_competition.features.Interface import FeatureInterface
from pandas import Series
from numpy import ndarray
import numpy as np
from tqdm import tqdm


class Feature(FeatureInterface):
    
    def __init__(self, name:str, subModule:str,
                 config: dict, txtLocation:str) -> None:
        super().__init__(name, subModule, config, txtLocation)
        # may takes approx 58 secs to download if 
        # this is the first time using
        
        
    def extract(self, documents:Series) -> ndarray:
        """given a list of document name,
        extract the features of the document
        in an ndarray

        Parameters
        ----------
        documents : pd.Series[PosixPath]
            A series of the PosixPath of the documents,
            should have been verified at the problem setting stage
            that all these documents should exist

        Returns
        -------
        ndarray
            [description]
        """
        return self.extractionChoice[self.aggregateMethod](documents)
    
    def extractMean(self, documents:Series) -> ndarray:
        
        output = []
        for eachDocument in tqdm(documents, 
                                 total=len(documents),
                                 leave=False,
                                 mininterval=1
        ):
            with open(eachDocument, 'r') as txtFile:
                text = txtFile.readlines()
                
            result = self.model.encode(text).mean(axis=0)
            output.append(result)
            
        output = np.array(output,dtype=float)
        return output   
    
    def extractOverall(self, documents:Series) -> ndarray:
        
        output = []
        for eachDocument in tqdm(documents, 
                                 total=len(documents),
                                 leave=False,
                                 mininterval=1
        ):
            with open(eachDocument, 'r') as txtFile:
                text = txtFile.read()
                
            result = self.model.encode(text)
            output.append(result)
            
        output = np.array(output,dtype=float)
        return output            
        