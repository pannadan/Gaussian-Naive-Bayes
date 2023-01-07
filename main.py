# Gaussian Naive Bayes Classifier used to Make predictions on Iris Dataset

import numpy as np
from random import randrange
import csv
import math


# Load a CSV file
def load_csv_dataset(filename):
    lines = csv.reader(open(filename, 'rb'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
        # converts the strings to float values
    return dataset


# mean of numbers returned
def mean(numbers):
    return np.mean(numbers)


# returns standard deveiation of the number set
def stdev(numbers):
    return np.std(numbers)


# returns sigmoid number
def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# Cross validation with n number of folds to train and test the data
def cross_validation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Returns the accuracy of the model
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluates algorithm using the cross validation split function
def evaluate(dataset, algorithm, n_folds, ):
    folds = cross_validation(dataset, n_folds)
    percentages = list()
    for fold in folds:
        training = list(folds)
        training.remove(fold)
        training = sum(training, [])
        testing = list()
        for row in fold:
            row_copy = list(row)
            testing.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(training, testing, )
        actual = [row[-1] for row in fold]
        accuracy = accuracy(actual, predicted)
        percentages.append(accuracy)
    return percentages


# Naive Bayes Component


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = list()
        separated[vector[-1]].append(vector)
    return separated



# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    summaries.pop()
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Make a prediction with Naive Bayes on Iris Dataset
filename = 'iris.csv'
dataset = load_csv_dataset(filename)
# convert class column to integers
#str_column_to_int(dataset, len(dataset[0]) - 1)
# fit model
model = summarize_by_class(dataset)
# define a new record
row = [5.7, 2.9, 4.2, 1.3]
# predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))
