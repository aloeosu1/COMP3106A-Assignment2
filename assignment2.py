# COMP3106A 
# Assignment 2

import math
import csv


def gaussian_probability(x, mean, std_deviation):
  return (1 / (std_deviation * math.sqrt(2 * math.pi))) * math.e * ((1/2) * ((x - mean)/std_deviation)**2)
   

def naive_bayes_classifier(dataset_filepath, patient_measurements):
  # dataset_filepath is the full file path to a CSV file containing the dataset
  # patient_measurements is a list of [temperature, heart rate] measurements for a patient

  # most_likely_class is a string indicating the most likely class, either "healthy", "diseased"
  # class_probabilities is a two element list indicating the probability of each class in the order [healthy probability, diseased probability]

  #reading csv file and storing it
  data = []
  with open(dataset_filepath, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      #add health/diseased, temperature, heart rate 
      data.append([row[0], float(row[1]), float(row[2])])



  return most_likely_class, class_probabilities