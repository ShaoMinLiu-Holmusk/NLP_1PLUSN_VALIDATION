
from pathlib import PosixPath
from modules.feature_competition.features.Interface import FeatureInterface
from pandas import Series
from numpy import ndarray
import numpy as np
from holbert import holbert
from tqdm import tqdm

from lib.Toolbox.loggerWrap import CSVwriter
from itertools import repeat

class Feature(FeatureInterface):
    
    def __init__(self, name:str, subModule:str,
                 config: dict, txtLocation:str) -> None:
        super().__init__(name, subModule, config, txtLocation)
        # takes approx 18 secs
        self.model = holbert(config['holbertConfig'])
        # self.model = holbert("../config/holbert/ExampleDefaultConfig.json")
        writer = CSVwriter(config['cache'])
        columns = zip(repeat('feature_'), range(86))
        columns = list(name+str(num) for name, num in columns)
        columns.append('filename')
        self.writer = writer('myholbertFeatureCache',
                             columns,False)
        
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
        output = []
        for eachDocument in tqdm(documents, 
                                 total=len(documents),
                                 leave=False,
                                 mininterval=1,
                                 ncols=50
        ):
            with open(eachDocument, 'r') as txtFile:
                text = txtFile.read()
                
            # processing takes so long
            result = self.model.processDocument(text)
            
            if self.config['logit']:
                mseLab = result.MseClassifier.mseLab.sparseProb.max(axis=0)
            else:
                mseLab = result.MseClassifier.mseLab.sparseProb > 0 # (sentenceNum, 86)
                # negationMask = result.MseClassifier.mseC_negation.sparseProb > 0
                # mseLab[negationMask] = ~mseLab[negationMask]
                mseLab = mseLab.any(axis=0)
            output.append(mseLab)
            mseLab = mseLab.tolist()
            mseLab.append(eachDocument.name)
            self.writer.writeRow(mseLab)
        if self.config['logit']:
            output = np.array(output,dtype=float)
        else:
            output = np.array(output,dtype=int)
        return output
            
            
        