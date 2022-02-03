
from os import access, name
from logs import logDecorator as lD 
from lib.Toolbox.readerWrap import readYASON
from pathlib import Path, PosixPath

script = Path(__file__)
moduleName = script.parent.stem
scriptName = script.stem
config = readYASON('../config/config.json')
logBase = config['logging']['logBase'] + f'.modules.{moduleName}.{scriptName}'

from typing import Union, List
from pydoc import locate
from tqdm import tqdm
from modules.feature_competition.problem import setupProblem
from sklearn import metrics
from collections import defaultdict
from pandas import Series, read_csv, DataFrame, merge
from numpy import ndarray


@lD.log(logBase + '.startCompetition')
def startCompetition(logger,
              problems: List[PosixPath],
              features: list,
              models: list,
              settings: dict,
              outputLocation:PosixPath) -> dict:
    # liveOutput = outputLocation/'liveGameBoard.csv'
    gameStats = []
    proBar = tqdm(problems, 
                  desc='Problem',
                  mininterval=1,
                  disable=len(problems)==1)
    '''
    with open(liveOutput, 'a') as file:
        
        file.write(f'problem,model,feature,f1,ROCauc,PRauc,precision,recall,accuracy')
        file.write('\n')
    '''
    
    if settings['modelToNewFeature']:
        ensembleFeatureFull = None
        featureCount = 0
    
    for eachProblem in proBar:
        proBar.set_postfix_str(eachProblem.name)
        logger.info(f'Starting {eachProblem.name}')
        # is a string to a csv Path
        try: 
            # X_train_doc is a pd.Series[PosixPath] to the txtFiles
            X_train_doc, X_test_doc, y_train, y_test = setupProblem(problemPath = eachProblem,
                                                        seed = settings['seed'],
                                                        testRatio = settings['testRatio'],
                                                        textLocation = settings['textLocation'],
                                                        negation = settings['negation'])
        except Exception as err:
            # wrong format for this csv, skip this csv
            if str(err) == 'FormatError':
                logger.warning(f'{eachProblem.name} is not in the right format')
                continue
            else:
                raise err
            
        proBar.set_postfix_str(eachProblem.name + ' -> Document loaded')
        for eachFeature in features:
            if not eachFeature['participating']:
                continue
            
            X_train = getFeatures(eachFeature, X_train_doc)
            X_test = getFeatures(eachFeature, X_test_doc)
            
            '''
            featurePath = f"modules.feature_competition.features.{eachFeature['subModule']}.Feature"
            Feature = locate(str(featurePath))
            if Feature == None:
                logger.warning(f"feature {eachFeature['name']} is not loaded")
                continue
            else:
                logger.info(f"feature {eachFeature['name']} is loaded")
            thisFeature = Feature(name = eachFeature['name'],
                                  subModule = eachFeature['subModule'],
                                  config = eachFeature['subModuleConfig'], 
                                  txtLocation = settings['textLocation'])
            logger.info(f"feature {eachFeature['name']} is initialised")
            
            
            X_train = thisFeature.extract(X_train_doc)
            X_test = thisFeature.extract(X_test_doc)
            '''
            
            for eachModel in models:
                if not eachModel['participating']:
                    continue
                if eachModel['onlyworksWith'] and not eachFeature['name'] in eachModel['onlyworksWith']:
                    # skip this model if this model doesnt work for this feature
                    # example, model `myholbertPred` only 
                    # works with myholbert features
                    continue
                proBar.set_postfix_str(eachProblem.name + f' -> {eachModel["name"]}')
                
                modelPath = f"modules.feature_competition.models.{eachModel['subModule']}.Model"
                Model = locate(str(modelPath))
                if Model == None:
                    logger.warning(f"model {eachModel['name']} is not loaded")
                    continue
                else:
                    logger.info(f"model {eachModel['name']} is loaded")
                    
                if eachModel['subModule'] == 'myholbertPredict':
                    # only special for `myholbertPredict`
                    thisModel = Model(eachModel['subModuleConfig'],
                                      problemName = eachProblem.name)
                else:
                    thisModel = Model(eachModel['subModuleConfig'])
                logger.info(f"Model: {eachModel['name']} initialised")
                thisModel.fit(X_train, y_train)
                y_pred_proba = thisModel.predict_proba(X_test)
                metrics = evaluate(y_test, y_pred_proba)
                
                
                gameStats.append([eachProblem.name, 
                             eachModel['name'], 
                             eachFeature['name'],
                             
                             metrics['f1'],
                             metrics['ROCauc'],
                             metrics['PRauc'],
                             metrics['precision'],
                             metrics['recall'],
                             metrics['accuracy']
                             ])
                '''
                with open(liveOutput, 'a') as file:
                    file.write(f"{eachProblem.name},{eachModel['name']},{eachFeature['name']},{metrics['accuracy']},{metrics['auc']},{metrics['f1']},{metrics['precision']},{metrics['recall']}")
                    file.write('\n')
                ''' 
                    
                # joning the new features
                if settings['modelToNewFeature']:
                    
                    
                    y_pred_proba_train = Series(thisModel.predict_proba(X_train))
                    y_pred_proba_test = Series(thisModel.predict_proba(X_test))
                    y_pred_proba = y_pred_proba_train.copy().append(y_pred_proba_test)
                    y_pred_proba = y_pred_proba.reset_index(drop=True)
                    y_pred_proba_index = X_train_doc.copy().append(X_test_doc)
                    y_pred_proba_index = y_pred_proba_index.apply(lambda ppath: ppath.name)
                    y_pred_proba_index = y_pred_proba_index.reset_index(drop=True)
                    ensembleFeature = DataFrame({'filename':y_pred_proba_index,
                                                 f'feature_{featureCount}':y_pred_proba})
                    featureCount += 1
                    ensembleFeature = ensembleFeature.drop_duplicates('filename')
                    if isinstance(ensembleFeatureFull, type(None)):
                        ensembleFeatureFull = ensembleFeature
                    else:
                        ensembleFeatureFull = merge(ensembleFeatureFull,
                                                    ensembleFeature,
                                                    how='outer',
                                                    on='filename')
    if settings['modelToNewFeature']:
          ensembleFeatureFull.to_csv(str(outputLocation/'emsembleFeatures.csv'))
    return gameStats

@lD.log(logBase + '.moduleRun')
def evaluate(logger,
             y_test, y_pred_proba):
    
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_proba, pos_label=True)
    PRauc = metrics.auc(recall, precision)
    
    y_pred = y_pred_proba > 0.5
    f1 = metrics.f1_score(y_test, y_pred)
    
    accuracy =  sum(y_test == y_pred)/ len(y_pred)
    
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    
    ROCauc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    return dict(
        f1 = f1,
        ROCauc = ROCauc,
        PRauc = PRauc,
        precision = precision,
        recall = recall,
        accuracy = accuracy
    )
    
@lD.log(logBase + '.getFeatures')  
def getFeatures(logger, 
                featureConfig:dict, 
                X_train_doc:Series)->ndarray:
    """Access the feature.csv
    filter using the X_train_doc
    sort it in the proper order

    Parameters
    ----------
    featureConfig : dict
        featureConfig[featurePath] -> some .csv contains `filename` 
        column, the rest are all features, no index column
    X_train_doc : Series[PosixPath]
        [description]

    Returns
    -------
    ndarray
        feature in numericals
    """
    featureCSV = read_csv(featureConfig['featurePath'])
    if 'Unnamed: 0' in featureCSV.columns:
        featureCSV = featureCSV.drop(columns='Unnamed: 0')
    featureCSV['filename'] = featureCSV['filename'].apply(lambda x: Path(x).stem)
    X_train_doc = X_train_doc.apply(lambda x: Path(x).stem)
    X_train_doc = X_train_doc.reset_index(drop=True) # reset index to numerical order
    # isolate the new index out as order
    # convert to a DataFrame
    X_train_doc = X_train_doc.reset_index(drop=False)
    X_train_doc.columns = ['order', 'filename']
    
    featureCSV = merge(featureCSV, X_train_doc,
                       how='right',
                       left_on='filename',
                       right_on='filename')
    # print(featureCSV.head())
    featureCSV = featureCSV[~featureCSV.order.isna()]
    featureCSV = featureCSV.sort_values('order')
    featureCSV = featureCSV.drop(columns=['order', 'filename'])    
    featureCSV = featureCSV.values
    # print(featureCSV)
    assert featureCSV.shape[0] == X_train_doc.shape[0], 'Number of rows are different'
    return featureCSV
    