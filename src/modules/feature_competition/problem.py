from ast import Raise
from multiprocessing import Condition
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path, PosixPath

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from typing import Union
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial

@lD.log(logBase + '.getActiveProblems')
def getActiveProblems(logger, problemCandidates: Union[str, list])->list:
    """Given a list of problems or a path to a folder, return a list
    of elligible problems. 
    (exist in folder, is a csv, participating=True)

    Parameters
    ----------
    problemCandidates : Union[str, list]
        Could be a string, that would a path to a folder, by default,
        all documents in this folder will participate in the competition
        
        If input happens to be a list, then its a list of dict, in the
        following format:
        {
            directory       : pathStr,
            participating   : bool
        }
        
        All input data participating must be a `*.csv` file

    Returns
    -------
    list
        A list of `*.csv` path that is going to participate in the problem
        But not all the csv may have the appropriate format. They need 
        to be further checked.  

    Raises
    ------
    ValueError
        Folder do not exist or possible no eligible problem exist 
        at all
    """

    if isinstance(problemCandidates, str):
        # if it is a string, read all the files from the directory
        problems = Path(problemCandidates)
        if not problems.exists():
            logger.error(f'Directory `problems` not found, end competition')
            raise ValueError(f'Directory `problems` not found')
        problems = list(path for path in problems.iterdir() if path.suffix == '.csv')
    elif isinstance(problemCandidates, list):
        problems = []
        for prob in problemCandidates:
            if not prob['participating']:
                continue
            elif not prob['directory'].endswith('.csv'):
                logger.warning(f"{prob['directory']} is not a `*.csv`")
                continue
            elif not Path(prob['directory']).exists():
                # check if each file exists
                logger.warning(f"{prob['directory']} do not exist")
                continue
            else:
                problems.append(Path(prob['directory']))
                
    if not problems:
        # if no problems exists
        logger.error(f'No problems are elligible, end competition')
        raise ValueError(f'No problems are elligible')
    
    logger.info(f'{len(problems)} are elligible for competition')
    return problems
   
   

 
@lD.log(logBase + '.setupProblem')
def setupProblem(logger,
                 problemPath: PosixPath,
                 seed: int,
                 testRatio: float,
                 textLocation:str,
                 negation:bool) -> tuple:
    """initialise the csv to a DataFrame, 
    * if there is a negation flag, invert in the label (ignore this for now)
    * remove rows where the document is not found
    * split the model into train/test and X Y

    Parameters
    ----------
    problemPath : PosixPath
        path to the csv file,
        header of csv: 
            ,user,document,label,negationFlag,historicalFlag,assertionType,annotationFlag
    seed : int
        random seed for splitting
    testRatio : float
        proportion of test
    textLocation: str
        str path where the original .txt files are stored

    Returns
    -------
    tuple[pd.Series]
        a tuple in the following:
        (X_train, X_test, y_train, y_test)
    """
    try:
        problemCSV = pd.read_csv(str(problemPath),
                                usecols = ['document', 'label', 'negationFlag'],
                                dtype={
                                    'document': str,
                                    'label': bool,
                                    'negationFlag': bool
                                })
    except Exception as error:
        if 'Usecols do not match columns' in str(error):
            raise Exception('FormatError')
        else:
            raise error
    
    if not isinstance(seed, int):
        logger.warning(f'Seed is not an integer, using seed = 2002')
        seed = 2002
    if not isinstance(testRatio, float) or not (0<testRatio<1):
        logger.warning(f'testRatio in incorrect format, using testRatio = 0.3')
        testRatio = 0.3
    
    # invert the negation flag
    if negation:
        negations = problemCSV[problemCSV.negationFlag].label
        problemCSV.loc[problemCSV.negationFlag, 'label'] = ~ problemCSV[problemCSV.negationFlag].label
    
    textLocation = Path(textLocation)
    docPosPath = partial(formDocumentPath, folder=textLocation)
    problemCSV.document = problemCSV.document.apply(docPosPath)
    documentExist = problemCSV.document.apply(lambda path:path.exists())
    docNotFound = sum(documentExist)
    if docNotFound:
        logger.warning(f'{docNotFound} documents are not found')
        problemCSV = problemCSV[documentExist]
    
    X_train, X_test, y_train, y_test = train_test_split(problemCSV.document, 
                                                        problemCSV.label, 
                                                        test_size=testRatio, 
                                                        random_state=seed,
                                                        stratify=problemCSV.label)
    return X_train, X_test, y_train, y_test

def formDocumentPath(documentName:str, folder:PosixPath)->PosixPath:
    return folder/ (Path(documentName).stem +'.txt')

def documentFound(documentName:str, folder:PosixPath)->bool:
    """[summary]

    Parameters
    ----------
    documentName : str
        name of document, just the name

    folder       : PosixPath
        location of the documents

    Returns
    -------
    bool
        [description]
    """
    return (folder/ (Path(documentName).stem +'.txt')).exists()

    