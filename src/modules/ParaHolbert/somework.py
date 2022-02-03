from time import sleep
from tqdm import tqdm
from holbert import holbert
import os

def doingWork(workAmount):
    processID = str(os.getpid())
    
    for i in tqdm(range(workAmount), 
                  disable=False,
                  desc=processID,
                  leave=False):
        sleep(1)
    return workAmount

def processDocuments(documents):
    print('booting holbert')
    model = holbert('../config/holbert/ExampleDefaultConfig.json')
    for eachSent in documents:
        model.processDocument(eachSent)
    return len(documents)

