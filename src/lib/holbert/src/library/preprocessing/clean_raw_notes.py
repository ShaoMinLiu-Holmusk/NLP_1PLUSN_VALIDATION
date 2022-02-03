import os, sys, nltk, re, pickle, json, shutil
import pandas as pd
# import hunspell
# import docx
import unicodedata
import scispacy, spacy 
import logging

from ..logWrapper import getLogger

loggerLevel = logging.WARNING
logger = getLogger(__name__, 
                   level=loggerLevel,
                   consoleLevel=loggerLevel, 
                   tofile=False) # do not put to file

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

try:
    spacy.load("en_core_sci_sm")
except OSError:
    logger.warning('en_core_sci_sm not install, running installation of corpus.')
    import pip
    pip.main(['install', 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz'])
    logger.warning('en_core_sci_sm successfully installed.')
    
    

# def doc_to_txt(input_dir, output_dir):
#     """Convert files with .doc extension to .txt

#     Args:
#         input_dir (str): path of directory containing the .doc file
#         output_dir (str): path of directory to contain the .txt file (file name unchanged, only extension + output folder changed)
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     for file in os.listdir(input_dir):
#         fn, extension = os.path.splitext(file)
#         if extension == '.docx':
#             dest_file_path = fn + '.txt'
#             write_text_file = open(os.path.join(output_dir, dest_file_path), "w+")
#             doc = docx.Document(os.path.join(input_dir, file))
#             for para in doc.paragraphs:
#                 write_text_file.write(para.text)
#                 write_text_file.write('\n')
#             write_text_file.close()
#     print('Converted .doc to .txt file')

def header_set(input_file):
    f = open(input_file, "r")
    _l = []
    for x in f:
        header = x.split(":")[0] 
        if header.isupper():
            _l.append(header)
    return _l

def identify_headers(input_file, tokenizer):
    f = open(input_file, "r")
    _dt = {
        'MENTAL STATUS EXAM'    : 'MSE',
        'MEDICATIONS'           : 'medication',
        'CURRENT MEDICATIONS'   : 'medication',
        'ASSESSMENT'            : 'diagnosis',
        'VITAL SIGNS'           : 'clinical measurement',
        'LAB WORK AND IMAGING'  : 'clinical measurement',
        'LAB WORK'              : 'clinical measurement'
    }
    sentence = []
    label = []
    for x in f:
        if x.isspace():
            continue
        header = x.split(":")[0] 
        temp = tokenizer.tokenize(x) 
        if header.isupper() and header in _dt: # check if first few words until colon is uppercase (i.e. header)
            sentence.extend([i for i in temp])
            label.extend([_dt[header] for i in range(len(temp))])
        else:
            sentence.extend([i for i in temp])
            label.extend([None for i in range(len(temp))])
    return sentence, label

def sentence_per_row(input_file, tokenizer, save=False):
    """Convert original clinical notes text file to a new text file with one sentence per row (to be used in Inception)

    Args:
        input_file (str): path to .txt file
        output_file (str): path to save .txt file
        tokenizer (nltk.tokenize.punkt): divides a text into a list of sentences; takes into account abbreviations
    """
    try:
        f = open(input_file, "r", encoding='utf8')
    except:
        try:
            f = open(input_file, "r", encoding='windows-1252')
        except:
            f = input_file
    if save:
        write_text_file = open(save, "w")

    allSentences = []
    for x in f:
        if x.isspace():
            allSentences.append('')
            if save:
                write_text_file.write('\n')
        else:
            try:
                sentence_l = tokenizer.tokenize(x)
            except:
                doc = tokenizer(x)
                sentence_l = [str(i) for i in list(doc.sents)]
            for i in sentence_l: # save one sentence per line in txt file
                if save:
                    if not i.isspace():
                        write_text_file.write(i)
                        write_text_file.write('\n')
                if not i.isspace():
                    allSentences.append(i)
    if save:
        write_text_file.close()

    return allSentences

def text_to_dataframe(input_file, tokenizer):
    try:
        f = open(input_file, "r", encoding='utf8')
    except:
        f = open(input_file, "r", encoding='windows-1252')
    sentence = []
    for x in f:
        if x.isspace():
            continue
        temp = tokenizer.tokenize(x)
        sentence.extend([unicodedata.normalize("NFKD", i) for i in temp]) # remove '\xa0' from string
    return sentence

## Peprocessing modules
def remove_dictation(text): 
    """Cleans clinical notes from dictation words

    Args:
        text (str): string to be preprocessed 

    Returns:
        str: string that has been preprocessed
    """
    text = str(text).strip()
    # text = re.sub(r"period.$", ".", text) # overcorrect it
    # text = re.sub(r"\bcomma,", ",", text)
    # text = re.sub(r"\bcomma\b", ",", text)
    # text = re.sub(r"#uh\b", "", text)
    # text = re.sub(r"#um\b", "", text)
    # text = re.sub(r"^first paragraph|^next paragraph|^new paragraph", "", text) # start of sentence
    # text = re.sub(r"\bend quote\b", "", text)
    # text = re.sub(r"\bquote\b|\bunquote\b", "", text)
    # text = re.sub(r"^next number", "", text)
    # text = re.sub(r"\betcetera", "etc", text)
    # text = text.strip(' ')
    return text

def keep_abbreviations(text, merge=True, save=None):
    '''This function applies regular expression to keep abbreviations and its subsequent word as one sentence
    to prevent it being tokenized into 2 sentences.

    For example, 
    "It is recommended the patient be continued on an Ativan 1 mg q.6 p.r.n. for acute anxiety." 
    will be split into
    "It is recommended the patient be continued on an Ativan 1 mg q.6 p.r.n.
    for acute anxiety."
    by the default tokenizer. (Tried scispacy and nltk punkt)
    
    Step 1- identify instances of abbreviations-word pairs to be kept as 1 sentence, 
            then merge them into 1 word by replacing spaces with __
    Step 2- revert back dummy __ into spaces

    Patterns:
    - Drug regimens, small letters (at least 2) separated by dots (abbreviations) followed by a small letter. e.g. "q.h.s. for"
    - Numbered list, numbers (max 2 digits) followed by a dot and space(s) and then a capital letter. e.g. "14. Ativan"

    Input
    ---
    text : str, either text or filename
    merge : bool, True= perform the first step, False=perform the 2nd step
    save: str, name of output file to save output (default=None)

    Output
    ---
    str which has been processed
    '''

    try:
        try:
            text = open(text)
            text = text.read()
        except:
            pass
        text = text.replace('\xa0', ' ')#.replace('\n', ' ')
            # text = ' '.join([i for i in text.split()])

        if merge:
            pattern = re.compile(r'(([a-zA-Z]\.)([a-zA-Z]\.)+\s+[a-z]+)|(\n[\d]{1,2}\.\s+[A-Z])')
            if save:
                with open(save, "w") as fid:
                    # print(pattern.sub(lambda m: (m.group(0).replace(' ', '__')), text), file=fid)
                    _towrite = pattern.sub(lambda m: (m.group(0).replace(' ', '__')), text)
                    fid.write(_towrite)
                
            # return pattern.sub(lambda m: (m.group(0).replace(' ', '__')), text)
            return _towrite

        else:
            pattern = re.compile(r'(([a-zA-Z]\.)([a-zA-Z]\.)+[__]+[a-z]+)|(^[\d]{1,2}\.[__]+[A-Z])')
            return pattern.sub(lambda m: (m.group(0).replace('__', ' ')), text)

    except Exception as e:
        print('Exception', e)

def expand_abbreviations(dic, text):
    """Expand medical abbreviation into its long form

    Args:
        dic (str): dictionary of abbreviations to be expanded
        text (str): string to be preprocessed

    Returns:    
        str: string that has been preprocessed
    """
    # 2021.03.27 Modified the regex pattern : 
    # 1) enclose each key in bracket and word boundary: (\b[KEY]\b)
    # 2) escape the regex, not the dictionary
    # This fixed previous problem : the first and last keys were not contained in one phrase (e.g. CABG matches ABG\b)
    pattern = re.compile("|".join([r"(\b"+re.escape(i)+r"\b)" for i in dic.keys()]), re.IGNORECASE) 
    text = pattern.sub(lambda m: dic[(m.group(0).upper())], text)
    return text

def correct_spelling(spellchecker, text):
    """For each word in the sentence (text), hunspell checks whether word exists in dictionary. If word does not exist, hunspell offers a suggestion.

    Args:
        spellchecker (hunspell spellchecker): hunspell.HunSpell('config/modules/vocabularies/en_US.dic', 'config/modules/vocabularies/en_US.aff')
        text (str): text to check for incorrectly spelled words (sentence)

    Returns:
        str: text with spelling corrected
    """ 
    words = nltk.word_tokenize(text)
    s = ''
    for w in words: 
        if not w.isalnum() and '-' not in w: # if not all characters are alphanumeric
            s = s+w if w != '[' else s+' '+w
        else:
            # ok = spellchecker.spell(w)
            ok = True
            if not ok:
                suggestions = spellchecker.suggest(w)
                try: # check if word is present in dt --> add count, else add dt
                    w = suggestions[0] # {(original word, corrected word): count}
                except:
                    w = w # might not have a suggestion
                s = s+' '+w
            else:
                s = s+' '+w
    s = s.replace('[ ', '[')
    return re.sub(r'(?<=[.,])(?=[^\s])', r' ', s.lstrip()) # add space after comma or dot

def add_medical_terms(medical_terms_text, hunspell_dic, hunspell_aff):
    '''This function load Hunspell instance and adds custom dictionary / vocab to the instance
    
    Args:
        medical_terms_text (path to text file with custom dictionary)
        hunspell_dic (path to default Hunspell dictionary)
        hunspell_aff (path to default Hunspell aff)

    Returns
        Modified Hunspell instance

    '''
    add_to_dict = [] 
    with open(medical_terms_text, "r") as text_file:
        for item in text_file:
            add_to_dict.append(item[:-1]) # remove line breaks
    spellchecker = hunspell.HunSpell(hunspell_dic, hunspell_aff)
    for i in add_to_dict:
        spellchecker.add(i)
    return spellchecker

## Load dataset
def read_text_file(input_file):
    """Converts text file to a list of sentences

    Args:
        text_file_name (str): a .txt file dataset

    Returns:
        list: a list of strings
    """
    f = open(input_file, "r")
    new_list = []
    for x in f:
        new_list.append(x)
    return new_list

## Save dataset
def save_text_file(text_list, output_file):
    write_text_file = open(output_file, "w+")
    for i in text_list: # save one sentence per line in txt file
        write_text_file.write(i)
        write_text_file.write('\n')
    write_text_file.close()

class preprocess:
    
    def __init__(self, config):
        self.config = config
        
        self.abbrev = dict(pd.read_csv(config['abbreviations']).values)
        # self.spellchecker = add_medical_terms(config['medical_terms_text'], config['hunspell_dic'], config['hunspell_aff'])
        # self.tok = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tok = spacy.load("en_core_sci_sm")
    
    def split(self, input_file, save=False):
        """Creates clinical notes that are one sentence per row from original clinical notes

        Args:
            input_file (str): path for clinical notes text file
                              Or 
                              input string for clinical notes text
            output_file (str): path to save clinical notes text file
        """
        try:
            sentences = sentence_per_row(input_file, self.tok, save=save)
            # print('Converted text file into one sentence per row.')
            return sentences
        except Exception as e:
            print(f'ERROR: Failed running preprocess.{sys._getframe().f_code.co_name}. ERROR: {e}')

    def run(self, 
            input_file,
            output_file = None):
        """Preprocess clinical notes

        Steps:
        1) Get list of sentences from .txt file
        2) Remove extra words from dictation 
        3) Expand abbreviations
        4) Correct spellings with modified Hunspell

        Args:
            input_file (str): path for clinical notes text file
                              Or 
                              input string for clinical notes text
            output_file (str): path to save clinical notes text file
                                If None, then only return the processed sentence list, 
                                and do not write to file
        """
        textInput = not os.path.exists(input_file)
        if textInput:
            itmd_dir = self.config['tempCache']
            if not os.path.exists(itmd_dir): os.makedirs(itmd_dir)
            with open(os.path.join(itmd_dir, 'input.txt'), 'w+') as file:
                file.write(input_file)
                input_file = os.path.join(itmd_dir, 'input.txt')
        try:

            # process from .txt
            itmd_dir = os.path.join(os.path.dirname(input_file), 'temp')

            # Prevent certain abbreviations pattern to be split into 2 sentences
            itmd_file = os.path.join(itmd_dir, '{}_temp.txt'.format(os.path.basename(input_file).replace('.txt','')))
            if not os.path.exists(itmd_dir): os.makedirs(itmd_dir)
                
            # return a set of string that has been processed to keep abbriviations
            text_list = keep_abbreviations(input_file, save=itmd_file)

            # 1. Split notes into sentences
            input_file = itmd_file
            itmd_file2 = os.path.join(itmd_dir, '{}_split.txt'.format(os.path.basename(input_file).replace('.txt','')))

            text_list = self.split(
                input_file, 
                save=itmd_file2) # create text files of one sentence per row

            # Revert back the abbreviations ("q.h.s.__for" --> "q.h.s. for")
            text_list = [keep_abbreviations(i, merge=False) for i in text_list]

            # 2. Remove leading, trailing, and double spaces
            text_list = [' '.join(i.strip().split()) for i in text_list]

            # # 1. Get sentences
            # text_list = read_text_file(input_file)

            # # 2. Preprocessing - remove dictation specific words
            # text_list = [remove_dictation(x) for x in text_list]

            # 3. Preprocessing - expand abbreviations to its long form
            text_list = list(expand_abbreviations(self.abbrev, x) for x in text_list)
            # # 4. Preprocessing - Apply hunspell to correct incorrectly spelled words
            # text_list = [correct_spelling(self.spellchecker, x) for x in text_list]

            # Save cleaned notes
            if output_file:
                save_text_file(text_list, output_file)

            # shutil.rmtree(itmd_dir)

            return text_list

        except Exception as e:
            print(f'ERROR: Failed running preprocess.{sys._getframe().f_code.co_name}. ERROR: {e}')

if __name__ == "__main__":
    config = json.load(open('../config/modules/preprocessing.json', 'r'))
    p = preprocess(config)

    input_dir = '../data/raw_data/04052021_batch_3/text_files'
    output_dir = '../data/intermediate/preprocessed/Batch3'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    frames = []
    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            try:
                input_file = os.path.join(input_dir, file)
                output_file = os.path.join(output_dir, file.replace('[','').replace(']',''))
                filename, file_extension = os.path.splitext(input_file)
                sentence = p.run(input_file, output_file)

                # create dataframe of sentences
                df = pd.DataFrame({'sentence': sentence})
                df['note_id'] = file.replace('[','').replace(']','')
                frames.append(df)
            except Exception as e:
                print(f'ERROR: Unable to preprocess this file: {file}. Error message: {e}')

    result = pd.concat(frames)
    result.to_csv('../data/intermediate/preprocessed/Batch3.csv')
