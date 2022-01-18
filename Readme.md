# NLP_1PLUSN_VALIDATION

This project is meant to be a validation of the 1PLUSN concept. In short, in order to make the models adaptive to changes in labels. A multi-label classification problem can be treated as multiple single-label prediction problem.  

This project will use the annotated shAlp as gound truth. (For example, identify text that are related to `suicidal thoughts`, create a train/test set from here).
Holbert will be used to predict the target, which will form the baseline.   

In addtional, other features can be extracted/transformed to generate richer features that aims at beating the baseline. This project is to produce a possibly better set of feature that is suitable for a variety of tasks. Thus, the features should be as model-agnostic as possible. Hence the same problem might be tackled using different models and different feature candidates.  

Feature Candidates:  
* Holbert
* Sbert
* Holbert + Sbert
* others

Problems:
* Suicidal Thought Prediction

Model:
* Random Forest
* Neural Network


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

Clone the repo using the following command, make sure:
* you have Python3.8 installed
* you are on dev branch for development
* you have AWS CLI installed 
* has access to s3://cliniciannotes-ds/package/
* set up main-DS profile

```
git@github.com:ShaoMinLiu-Holmusk/NLP_1PLUSN_VALIDATION.git  
make firstRun
```   


## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

## Installing

The folloiwing installations are for \*nix-like systems. 

For installation, first close this repository, and generate the virtual environment required for running the programs. 

This project framework uses [venv](https://docs.python.org/3/library/venv.html) for maintaining virtual environments. Please familiarize yourself with [venv](https://docs.python.org/3/library/venv.html) before working with this repository. You do not want to contaminate your system python while working with this repository.

A convenient script for doing this is present in the file [`bin/vEnv.sh`](../master/bin/vEnv.sh). This is automatically do the following things:

1. Generate a virtual environment
2. activate this environment
3. install all required libraries
4. deactivate the virtual environment and return to the prompt. 

At this point you are ready to run programs. However, remember that you will need to activate the virtual environment any time you use the program.

For activating your virtual environment, type `source env/bin/activate` in the project folder in [bash](https://www.gnu.org/software/bash/) or `source env/bin/activate.fish` if you are using the [fish](https://fishshell.com/) shell.
For deactivating, just type deactivate in your shell

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

 - Python 3.6

## Contributing

Please send in a pull request.

## Authors

ShaoMinLiu - Initial work (2022)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - Hat tip to anyone who's code was used
 - Inspiration
 - etc.
 