from logs import logDecorator as lD 
import jsonref, pprint
import neuroblu_postgres as nb
import pandas as pd

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.report_generation.report_generation'


@lD.log(logBase + '.generate_table')
def generate_table(logger, cohort1_tab, cohort2_tab):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''
    desc_table = nb.generate_descriptives(dataframe=[cohort1_tab, cohort2_tab], use_cols=['race', 'age','gender', 'marital_status', 'years_in_education', 'employment_status', 'CGIS-change'], cohort_names=['fluoxetine', 'trazodone'])

    return desc_table

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
    print('Main function of report generation')
    print('='*30)
    
    study_table_demo = pd.read_csv('../data/intermediate/clean_df.csv')
    # Get the patient_ids for the two cohorts based on the drug of interest
    cohort1_tab = study_table_demo.loc[study_table_demo['drug'] == 'fluoxetine']
    cohort2_tab = study_table_demo.loc[study_table_demo['drug'] == 'trazodone']

    ## Create and save baseline characteristics table
    desc_df = generate_table(cohort1_tab, cohort2_tab)
    desc_df.to_csv('../results/desc_table.csv')
    print('The baseline characteristics table is generated\n')

    print('Getting out of report generation')
    print('-'*30)

    return 

