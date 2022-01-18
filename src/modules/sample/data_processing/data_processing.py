from logs import logDecorator as lD 
import jsonref, pprint
import pandas as pd
import neuroblu_postgres as nb

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.data_processing.data_processing'


@lD.log(logBase + '.add_age')
def add_age(logger, table):
  # Get patient cohort's age data
  yob = nb.get_birth_year(table['person_id'], dbname='cdm')
  tab = pd.merge(yob, table, on = 'person_id')
  tab['drug_index_start_date'] = pd.to_datetime(tab['drug_index_start_date'])
  tab['age'] =  tab['drug_index_start_date'].dt.year - tab['birth_year']
  tab = tab.drop(['birth_year'], axis=1)
  return tab

@lD.log(logBase + '.add_demographics')
def add_demographics(logger, table):
  # Get patient cohort's race data
  demo = nb.get_demographics(table['person_id'], dbname='cdm')
  demo = demo[['person_id', 'race', 'gender', 'marital_status', 'employment_status', 'years_in_education']]
  tab = pd.merge(demo, table, on = 'person_id')
  return tab

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for data_processing
    
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
    print('Main function of data processing')
    print('='*30)

    study_table = pd.read_csv('../data/raw_data/cohort.csv')
    study_table_demo = add_age(study_table)
    study_table_demo = add_demographics(study_table_demo)

    # Retain only the fields that are required for analysis
    study_table_demo = study_table_demo[['person_id', 'race', 'age', 'gender', 'marital_status', 'years_in_education', 'employment_status', 'cgis_start', 'cgis_end', 'drug']]

    # Define change in CGIS as CGIS at start of the prescription minus CGIS at the
    # end of the prescription
    study_table_demo['CGIS-change'] = study_table_demo['cgis_end'] - study_table_demo['cgis_start']
    study_table_demo.to_csv('../data/intermediate/clean_df.csv')

    print('Getting out of data processing')
    print('-'*30)

    return 

