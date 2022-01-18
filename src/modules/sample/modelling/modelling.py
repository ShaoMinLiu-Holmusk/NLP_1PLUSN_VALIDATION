from logs import logDecorator as lD 
import jsonref, pprint
import matplotlib.pyplot as plt
import pandas as pd
import os

config = jsonref.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.modelling.modelling'

@lD.log(logBase + '.cgi_change')
def cgi_change(logger, table):
  """
  Calculates the cgi-s change and compare the differences between the two cohorts 
  """
  # Set the style (optional)
  plt.style.use('neuroblu')

  # Set matplotlib font parameters
  plt.rc('axes', titlesize=16)     # fontsize of the axes title
  plt.rc('axes', labelsize=16)     # fontsize of the x and y labels
  plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
  
  table = table[['drug', 'CGIS-change']]
  
  drug1, drug2 = table['drug'].unique().tolist()
  
  drug1_data = table.loc[table['drug'] == drug1]
  drug2_data = table.loc[table['drug'] == drug2]
  
  d1 = drug1_data['CGIS-change'].value_counts(normalize=True)*100
  d2 = drug2_data['CGIS-change'].value_counts(normalize=True)*100
  
  width = 0.35
  plt.figure(figsize=(6,6))
  plt.bar(d1.index - width/2, d1.values, width, label= drug1)
  plt.bar(d2.index + width/2, d2.values, width, label= drug2)
  plt.legend(loc='upper left', prop={'size': 16})
  plt.xlabel('CGIS-change from Baseline')
  plt.ylabel('Percentage of patients (%)')
  plt.tight_layout()

  plt.savefig('../results/study_results.png')
  return 

@lD.log(logBase + '.main')
def main(logger, resultsDict):
    '''main function for modelling
    
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
    print('Main function of modelling')
    print('='*30)
    
    table = pd.read_csv('../data/intermediate/clean_df.csv')
    cgi_change(table)

    print('Getting out of modelling')
    print('-'*30)

    return 

