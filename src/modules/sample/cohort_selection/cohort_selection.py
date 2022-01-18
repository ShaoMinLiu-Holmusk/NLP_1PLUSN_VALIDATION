from logs import logDecorator as lD 
import jsonref
import neuroblu_postgres as nb
import pandas as pd

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.cohort_selection.cohort_selection'
study_config = jsonref.load(open('../config/modules/sample_study_config.json'))

def get_mdd_codes():
    mdd_codes = ["296.2", "296.21", "296.22", "296.23", "296.24", "296.25", "296.26", "296.3","296.31", "296.32", "296.33", "296.34", "296.35", "296.36", "F32.0", "F32.1","F32.2", "F32.3", "F32.4", "F32.5", "F32.81", "F32.89", "F32.9", "F33.0","F33.1", "F33.2", "F33.3", "F33.40", "F33.41", "F33.42", "F33.8", "F33.9"]

    return mdd_codes


@lD.log(logBase + '.get_age')
def get_age(logger, pid):
    # Get year of first visit
    pid_first_visit_date = nb.get_first_visit_date(patients=pid, dbname='cdm')
    pid_first_visit_date['first_visit_year'] = pd.to_datetime(pid_first_visit_date['first_visit_date']).dt.year
    
    # Get birth year
    mdd_patients_birth_year = nb.get_birth_year(patients=pid, dbname='cdm')
    merged_df = pd.merge(pid_first_visit_date, mdd_patients_birth_year, on='person_id')
    merged_df['age'] = merged_df['first_visit_year']- merged_df['birth_year']

    return merged_df[['person_id', 'age']]

@lD.log(logBase + '.get_test_MDD_patients')
def get_test_MDD_patients(logger):
    ''' 
    This function returns 500 arbitrary adult MDD patients
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    '''
    mdd_patients = nb.filter_patients_by_ICD(get_mdd_codes(), dbname='cdm')
    mdd_patients_age = get_age(mdd_patients)
    filter_adults = mdd_patients_age.loc[mdd_patients_age['age'] > 18]

    return filter_adults['person_id'][:500]

@lD.log(logBase + '._merge_cgis')
def _merge_cgis(logger, cgis_tab, cdd_tab, cdd_tab_date, window_days):
  tab = pd.merge(cgis_tab, cdd_tab, on = 'person_id')
  tab_filt = tab.loc[abs(tab['measurement_date'] - tab[cdd_tab_date]).dt.days < window_days]  
  tab_filt_val = round(tab_filt.groupby('person_id')['value'].mean())
  
  cdd_tab_filt = pd.merge(cdd_tab, tab_filt_val, on = 'person_id')
  return cdd_tab_filt

@lD.log(logBase + '.get_first_diagnosis')
def get_first_diagnosis(logger, patient_ids):
  # Get first diagnosis date of the cohort patients
  
  # Get the MDD ICD codes from the diagnosis category map
  mdd_codes = get_mdd_codes()

  diagnosis = nb.get_first_diagnosis_date(patients=patient_ids, icd_codes=mdd_codes, dbname='cdm')
  return diagnosis

@lD.log(logBase + '.get_index_drug')
def get_index_drug(logger, cohort_diagnosis_tab, drug_names, study_period):
  # Fetching the drug data from neuroblu database
  drug_table = nb.get_drug(cohort_diagnosis_tab['person_id'], dbname='cdm')
  
  # Retaining only the relevant drug information
  drug_table = drug_table.loc[drug_table['drug'].isin(drug_names)]
  
  # Join drug and cohort table on `person_id`
  drug_diagnosis_tab = pd.merge(drug_table, cohort_diagnosis_tab, on='person_id')
  
  # Filter out rows with drug prescription before diagnosis
  drug_diag_filt = drug_diagnosis_tab.loc[drug_diagnosis_tab['first_diagnosis_date'] <= drug_diagnosis_tab['start_date']]
  
  patient_drug_table = drug_diag_filt.sort_values(["person_id", "start_date"],
                          ascending = (False, True))
  patient_drug_table = patient_drug_table.drop_duplicates(['person_id', 'drug'], keep='first') 
  
  # Identify patients who are on multiple drugs
  n_drugs_prescribed = patient_drug_table.groupby('person_id').size().sort_values()
  patient_mul_drugs = n_drugs_prescribed[n_drugs_prescribed > 1].index.tolist()
  
  # Remove patients who are on multiple drugs
  patients_single_drug = patient_drug_table.loc[~patient_drug_table['person_id'].isin(patient_mul_drugs)]
  
  # Cast the date as datetime object for date subtraction
  patients_single_drug['end_date'] = pd.to_datetime(patients_single_drug['end_date'])
  patients_single_drug['start_date'] = pd.to_datetime(patients_single_drug['start_date'])
  
  # Filter out patients who have spend less time than the `study_period`
  filt_tab = patients_single_drug.loc[(patients_single_drug['end_date'] - 
                          patients_single_drug['start_date']).dt.days > study_period]
    
  # Rename the columns of the table 
  filt_tab.rename({'start_date': 'drug_index_start_date',
                   'end_date': 'drug_index_end_date'},
                  axis=1, inplace=True)
  return filt_tab
  
@lD.log(logBase + '.get_cgis')
def get_cgis(logger, cohort_diagnosis_drug_tab, window_days, min_cgis):
  cgis_table = nb.get_smooth_CGIS(cohort_diagnosis_drug_tab['person_id'], dbname='cdm')
  
  # Cast the measurement_date as datetime object for date subtraction
  cgis_table['measurement_date'] = pd.to_datetime(cgis_table['measurement_date'])
  
  # Get cgis for the start date
  cohort_diagnosis_drug_tab = _merge_cgis(cgis_table, cohort_diagnosis_drug_tab, 'drug_index_start_date', window_days)
  
  # Filter patients with index CGIS < `min_CGI`
  cohort_diagnosis_drug_tab = cohort_diagnosis_drug_tab.loc[cohort_diagnosis_drug_tab['value'] >= min_cgis]

  cohort_diagnosis_drug_tab.rename({'value': 'cgis_start'},
                                   axis=1, inplace=True)
  
  # Get cgis for the end date
  cohort_diagnosis_drug_tab = _merge_cgis(cgis_table, cohort_diagnosis_drug_tab, 'drug_index_end_date', window_days)
  cohort_diagnosis_drug_tab.rename({'value': 'cgis_end'},
                                   axis=1, inplace=True)
  
  return cohort_diagnosis_drug_tab
  
def get_cohort_study(config):
  drug_name1 = config['drug1']
  drug_name2 = config['drug2']
  window_days = config['win_days']
  study_period = config['study_period']
  min_index_cgis = config['min_CGI']
  
  # Get the cohort patient ids
  patient_ids = get_test_MDD_patients()
  
  drug1_ids = nb.filter_patients_by_drug(drug_name1, dbname='cdm')
  drug2_ids = nb.filter_patients_by_drug(drug_name2, dbname='cdm')
  patient_ids_drugs = list(set(patient_ids).intersection(set(drug1_ids).union(drug2_ids)))
  
  print('Fetching diagnosis')
  cohort_diag = get_first_diagnosis(patient_ids_drugs)
  
  print('Fetching drug information')
  cohort_diag_drug = get_index_drug(cohort_diag,
                                    (drug_name1, drug_name2),
                                    study_period)
  
  print('Fetching CGI-S at the start & end-points') 
  cohort_diag_drug_cgis = get_cgis(cohort_diag_drug, window_days, min_index_cgis)
  
  return cohort_diag_drug_cgis

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

    study_table = get_cohort_study(study_config)
    study_table.to_csv('../data/raw_data/cohort.csv')

    print('Getting out of cohort selection')
    print('-'*30)

    return 

