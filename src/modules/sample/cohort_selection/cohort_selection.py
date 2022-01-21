from logs import logDecorator as lD 
import jsonref, pprint

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.cohort_selection.cohort_selection'


@lD.log(logBase + '.doSomething')
def doSomething(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''

    print('We are in cohort_selection')

    return

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for cohort_selection
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dictionary containing information about the 
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    '''

    print('='*30)
    print('Main function of cohort selection')
    print('='*30)
    print('We get a copy of the result dictionary over here ...')
    pprint.pprint(resultsDict)

    doSomething()

    print('Getting out of cohort selection')
    print('-'*30)

    return

