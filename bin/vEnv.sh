#!/bin/bash

#----------------------------------------------
# Note that this is the standard way of doing 
# things in Python 3.6. Earlier versions used
# virtulalenv. It is best to convert to 3.6
# before you do anything else. 
# Note that the default Python version in 
# the AWS Ubuntu is 3.5 at the moment. You
# will need to upgrade the the new version 
# if you wish to use this environment in 
# AWS
#----------------------------------------------
python3.8 -m venv env

# this is for bash. Activate
# it differently for different shells
#--------------------------------------
source env/bin/activate 

pip3 install --upgrade pip

if [ -e requirements.txt ]; then

    pip3 install -r requirements.txt
    
else

    pip3 install pytest
    pip3 install pytest-cov
    pip3 install sphinx
    pip3 install sphinx_rtd_theme

    # Logging into logstash
    pip3 install python-logstash

    # networkX for graphics
    pip3 install networkx
    pip3 install pydot # dot layout
    
    # Utilities
    pip3 install jupyter
    pip3 install jupyterlab
    pip3 install tqdm
    pip3 install jsonref

    # scientific libraries
    pip3 install numpy
    pip3 install scipy
    pip3 install pandas

    # ML libraries
    pip3 install -U scikit-learn
    
    # database stuff
    pip3 install psycopg2-binary

    # Charting libraries
    pip3 install matplotlib
    pip3 install seaborn

    # Neuroblu python package
    pip3 install -e git+ssh://git@github.com/Holmusk/neuroblu_postgres.git@v0.7.0#egg=neuroblu_postgres

    # Sbert
    pip3 install sentence-transformers

    pip3 freeze > requirements.txt

    # Generate the documentation
    cd src 
    make doc
    cd ..

fi

# download Holbert
aws s3 cp s3://cliniciannotes-ds/package/ ./HolbertPackage/ --recursive --profile main-DS
pip3 install HolbertPackage/holbert-0.0.1-py3-none-any.whl


# Download all models from S3 (specified in download.txt)
# Note: The second column specifies the path to which the model will be saved. 

# Models include:
# 1) BioDischargeSummaryBERT model: config/pretrained/biobert_pretrain_output_disch_100000
# 2) Sentence topic classifier: results/sentenceClassifier_V2/20210713_153829_20210713_202132/model0
# 3) MSE IE model: results/MseAllTokens/20210804_030119/20210804_030119/model2
# 4) Stressor IE model: results/StressorsAllTokens/20210909_161211/20210911_171841/model0

input='download.txt'
while IFS= read -r line; 
  do 
  aws s3 sync ${line}; 
done < "$input"


deactivate