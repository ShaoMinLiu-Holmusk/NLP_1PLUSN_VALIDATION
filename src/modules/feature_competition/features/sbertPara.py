
from array import array
from pathlib import PosixPath
from modules.feature_competition.features.Interface import FeatureInterface
from pandas import Series
from numpy import ndarray
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pyspark import SparkContext, SparkConf

# trying to make the feature extraction 
# parallel using pyspark
# not working well so far
# maybe should try using multi-processing
# at least can still visualise process via tqdm

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def extractOverall(documents:Series) -> ndarray:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunkResult = []
    for documentPath in tqdm(documents, len(documents)):
        with open(documentPath, 'r') as txtFile:
            text = txtFile.read()
        result = model.encode(text)
        chunkResult.append(result)    
    return chunkResult  
        
class Feature(FeatureInterface):
    
    def __init__(self, name:str, subModule:str,
                 config: dict, txtLocation:str) -> None:
        super().__init__(name, subModule, config, txtLocation)
        # may takes approx 58 secs to download if 
        # this is the first time using
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.aggregateMethod = config['aggregateMethod']
        self.extractionChoice = {
            'mean'      : self.extractMean
        }
        sc_conf = SparkConf()
        sc_conf.set('spark.ui.showConsoleProgress', True)
        self.sc = SparkContext(master="local[2]", conf=sc_conf)
        
        
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
        RDD = self.sc.parallelize(chunks(documents, 8))
        RDD = RDD.map(extractOverall) # RDD returned
        RDDresult = RDD.collect()
        result = []
        for i in RDDresult:
            result.extend(i)
        result = array(result)
        return result
    
    def extractMean(self, documentPath:Series) -> ndarray:
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(documentPath, 'r') as txtFile:
            text = txtFile.readlines()
                
        result = self.model.encode(text).mean(axis=0)
        return result