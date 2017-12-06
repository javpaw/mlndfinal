import os
import codecs
import csv
from itertools import islice
from clean_data import text_to_wordlist
from datatypes import TrainingData
from datatypes import TestingData

def check_path(path):
    if not os.path.isfile(path):
        raise Exception("file path %s doesn't exists"%path)
    return path;

def get_file(path):
    return codecs.open(path, encoding = 'utf-8')

def get_file_iterator(fl):
    return csv.reader(fl, delimiter=',')

def check_file_structure(reader, expected_structure):
    expected = ','.join(expected_structure)
    titles = ','.join(next(reader))
    if(titles != expected):
        raise Exception("The titles in the file are not the expected ones: %s, expected: %s", titles, expected)
    return reader;


def extract_training_information(reader, num_samples = None):
    Q1 = 3
    Q2 = 4
    LABEL = 5
    q1 = []
    q2 = []
    labels = []
    itr = reader if (num_samples is None) else islice(reader, num_samples) 
    for line in itr:
        q1.append(text_to_wordlist(line[Q1]))
        q2.append(text_to_wordlist(line[Q2]))
        labels.append(int(line[LABEL]))
    return TrainingData(q1, q2, labels)

def extract_testing_information(reader, num_samples = None):
    ID = 0
    Q1 = 1
    Q2 = 2
    tid = []
    q1 = []
    q2 = []
    itr = reader if (num_samples is None) else islice(reader, num_samples) 
    for line in itr:
        q1.append(text_to_wordlist(line[Q1]))
        q2.append(text_to_wordlist(line[Q2]))
        tid.append(int(line[ID]))
    return TestingData(q1, q2, tid)


def load_and_clean_data_factory(extractFn, csv_structure):
    def internal_load_and_clean(training_path, num_samples = None):
        path    = check_path(training_path);
        with get_file(path) as fl:
            readerWithHeader = get_file_iterator(fl)
            reader  = check_file_structure(readerWithHeader,csv_structure)
            results = extractFn(reader, num_samples)
        return results
    return internal_load_and_clean

training_structure = ["id","qid1","qid2","question1","question2","is_duplicate"]
testing_structure = ["test_id","question1","question2"]

load_and_clean_training_data = load_and_clean_data_factory(extract_training_information, training_structure)
load_and_clean_testing_data = load_and_clean_data_factory(extract_testing_information, testing_structure)

