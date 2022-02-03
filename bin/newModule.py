from email.policy import default
from pathlib import Path
import json
from tkinter import W
from argparse import ArgumentParser
from warnings import warn

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-n','--moduleName',required=True,type=str)
    args.add_argument('-o','--owner',type=str, default='')
    args.add_argument('-d','--description',type=str, default='')
    args = args.parse_args()
    moduleName = args.moduleName
    
    moduleFolder = Path(f'./modules/{moduleName}')
    while moduleFolder.exists():
        warning = f'Module `{moduleName}` already exist!'
        warn(warning)
        moduleName = input(f'Please rename: ')
        moduleFolder = Path(f'./modules/{moduleName}')
        
    moduleFolder.mkdir(parents=True)
    with open(moduleFolder / '__init__.py', 'w') as mainFile:
        mainFile.write("")
    
    mainTemplate = '''
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from lib.Toolbox.writerWrap import writeYASON
from pathlib import Path
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

@lD.log(logBase + '.main')
def main(logger, resultsDict)->None:


    outputLocation = Path(moduleConfig['output']['location']) / runID
    outputLocation.mkdir(parents=True, exist_ok=True)
    # keep a copy of the config file for regeneration
    writeYASON(moduleConfig, outputLocation/Path(config['configVersions'][moduleName]).name)
    pass

if __name__ == '__main__':
    main('resultsDict')
    '''
    with open(moduleFolder / 'main.py', 'w') as mainFile:
        mainFile.write(mainTemplate)
        
    # create config files
    configTemplate = f'''
# trying out using YAML as config file
# YAML supports commenting unlike JSON
# it is more user-friendly for documentation purposes
# other than this, yamlFile and JSON can both be read as
# a dictionary format, thus little changes are needed 
# to be made to the actual code

dataSource    : ../data/

output:
  location    : ../data/intermediate/{moduleName}
  posfix      : '' # give a tag to this run, appended to runID, default to empty
    '''
    configFolder = Path(f'../config/modules/{moduleName}')
    configFolder.mkdir(exist_ok=True, parents=True)
    with open(configFolder / 'sampleConfig(dev).yaml', 'w') as mainFile:
        mainFile.write(configTemplate)
        
    config = json.load(open('../config/config.json'))
    config['configVersions'][moduleName] = str(configFolder / 'sampleConfig(dev).yaml')
    with open('../config/config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)
        
    config = json.load(open('../config/modules.json'))
    config.append(
        {
        "moduleName" : moduleName,
        "path"       : f"modules/{moduleName}/main.py",
        "execute"    : False,
        "description": args.description,
        "owner"      : args.owner
    }
    )
    with open('../config/modules.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)
    
        
    
    