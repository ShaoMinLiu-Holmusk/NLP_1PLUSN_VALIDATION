import sys
sys.path.insert(0,'../src/')

from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from lib.Toolbox.writerWrap import writeYASON
from pathlib import Path, PosixPath
from datetime import datetime

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

moduleConfig = config['configVersions'][moduleName]
moduleConfig = readYASON(moduleConfig)
runID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
tag = moduleConfig['output']['posfix'] # each run has an timebased unique ID 
runID = '_'.join((runID,tag)) if tag else runID # ID appened with tag(optional)


from modules.feature_competition.problem import getActiveProblems
from modules.feature_competition.competition import startCompetition
import pandas as pd

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:
    """The main method for the competition module.
    The module config takes in different model architecture, 
    feature design and target problem.
    
    Each feature is used to solve different problems using different
    model. The best ideal feature should get the best scores in 
    all problem using any model. (problem and model agnostic)
    
    Since this is a binary problem, we are looking at accuracy.
    True Positive and False Negative solving the problem.
    
    The module will then return a report, that has the recorded 
    performance metric and a visualisation of the performances.  

    Parameters
    ----------
    resultsDict : [type]
        [description]
    """
    
    outputLocation = Path(moduleConfig['output']['location']) / runID
    # outputLocation.mkdir(parents=True, exist_ok=True)
    settings = moduleConfig['competitionSettings']
    
    problems = getActiveProblems(moduleConfig['problems'])
    
    outputLocation.mkdir(exist_ok=True, parents=True)
    gameStats = startCompetition(problems,
                               moduleConfig['features'],
                               moduleConfig['models'],
                               settings,
                               outputLocation)
    
    # keep a copy of the config file for regeneration
    writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    gameStats = pd.DataFrame(gameStats, columns=[
        'Problem', 'Model', 'Feature', 
        'F1','ROCauc','PRauc', 'Precision', 'Recall','Accuracy',
    ])
    gameStats.to_csv(outputLocation/'gameStats.csv')

if __name__ == '__main__':
    main('resultsDict')
    