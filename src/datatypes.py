from collections import namedtuple

TrainingData = namedtuple('TrainingData', ['q1', 'q2', 'labels'])
TestingData = namedtuple('TestingData', ['q1', 'q2', 'id'])