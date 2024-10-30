# Michael Han
# 101157504
# COMP3106A 
# Assignment 2

import math
import csv
import numpy as np
import scipy

#returns the gaussian distrbution
def gaussian_probability(x, mean, std):
  return (1 / (math.sqrt(2 * math.pi * (std**2)))) * (math.exp(-0.5 * ((x - mean) / std)**2))


'''

#returns the normal gaussian distrbution using scipy (gives same answer as my function)
def gaussian_probability(x, mean, std):
  return scipy.stats.norm.pdf(x, loc = mean, scale = std)

'''


#get stats needed for each class for calculating probabilities
def get_stats(data):
  #temperature and heart rate value lists
  temps = []
  hr = []

  #add temp and hr value to their lists
  for x in data:
    #add first element (temp) to temp list
    temps.append(x[0])
    #add second element (hr) to hr list
    hr.append(x[1])

  #get mean and standard deviation for temperatures
  temps_mean = np.mean(temps)
  temps_std = np.std(temps)

  #get mean and standard deviation for heart rates
  hr_mean = np.mean(hr)
  hr_std = np.std(hr)

  return temps_mean, temps_std, hr_mean, hr_std
   

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
  file.close()

  #seperate into healthy and diseased lists
  healthy = []
  diseased = []

  #for every row in data
  for row in data:
    #if healthy, append temp and HR to healthy list
    if row[0] == "healthy":
      healthy.append(row[1:])
    #if diseased, append temp and HR to diseased list
    elif row[0] == "diseased":
      diseased.append(row[1:])

  #print("Healthy: ", healthy)
  #print("Diseased: ", diseased)

  
  #getting prior probabilities
  total_patients = len(data)
  prior_healthy = len(healthy) / total_patients
  prior_diseased = len(diseased) / total_patients

  #print("prior probabilities: ", prior_healthy, prior_diseased)

  #reading patient_measurements.txt
  patient_data = []
  with open(patient_measurements, "r") as file:
    #read the file
    data2 = file.read()
    #get rid of [] brackets, split values at ,
    val = data2.strip("[]").split(",")
    patient_data = [float(val[0]), int(val[1])]

  file.close()

  #getting patient temperature and heart rate
  patient_temp, patient_hr = patient_data

  #test prints
  #print("patient temp: ", patient_temp)
  #print("patient heart rate: ", patient_hr)
  #print("~~~~~~~~~~~~~~~~\n")

  #getting stats needed to calculate probabilities for each class
  healthy_temp_mean, healthy_temp_std, healthy_hr_mean, healthy_hr_std = get_stats(healthy)
  diseased_temp_mean, diseased_temp_std, diseased_hr_mean, diseased_hr_std = get_stats(diseased)

  #test prints
  #print("Healthy Temp Mean: ", healthy_temp_mean)
  #print("Healthy Temp STD: ", healthy_temp_std)
  #print("Healthy HR Mean: ", healthy_hr_mean)
  #print("Healthy HR STD: ", healthy_hr_std)
  #print("Diseased Temp Mean: ", diseased_temp_mean)
  #print("Diseased Temp STD: ", diseased_temp_std)
  #print("Diseased HR Mean: ", diseased_hr_mean)
  #print("Diseased HR STD: ", diseased_hr_std)


  #getting likelihood probability for diseased class
  p_temp_healthy = gaussian_probability(patient_temp, healthy_temp_mean, healthy_temp_std)
  p_hr_healthy = gaussian_probability(patient_hr, healthy_hr_mean, healthy_hr_std)
  healthy_likelihood = p_temp_healthy * p_hr_healthy 

  #test prints
  #print("Healthy likelihoods: ")
  #print("P(temp|healthy): ", p_temp_healthy)
  #print("P(hr|healthy): ", p_hr_healthy)
  #print("Healthy Likelihood: ", healthy_likelihood)

  #getting likelihood probability for diseased class
  p_temp_diseased = gaussian_probability(patient_temp, diseased_temp_mean, diseased_temp_std)
  p_hr_diseased = gaussian_probability(patient_hr, diseased_hr_mean, diseased_hr_std)
  diseased_likelihood = p_temp_diseased * p_hr_diseased 

  #test prints
  #print("Diseased likelihoods: ")
  #print("P(temp|diseased): ", p_temp_diseased)
  #print("P(hr|diseased): ", p_hr_diseased)
  #print("Diseased Likelihood: ", diseased_likelihood)

  
  #getting marginal likelihood(evidence)
  marginal_likelihood = (healthy_likelihood * prior_healthy) + (diseased_likelihood * prior_diseased)


  #put in bayes formula and calculate posterior probability (healthy and diseased probabilities)

  healthy_probability = (healthy_likelihood * prior_healthy ) / marginal_likelihood
  diseased_probability = (diseased_likelihood * prior_diseased) / marginal_likelihood

  print("Healthy probability: ", healthy_probability)
  print("Diseased probability: ", diseased_probability)


  #comparing probabilites
  if healthy_probability > diseased_probability:
    return "healthy", [healthy_probability, diseased_probability]
  else:
    return "diseased", [healthy_probability, diseased_probability]


def main():
  #edit filepaths as needed
  most_likely_class, class_probabilities = naive_bayes_classifier("C:/Users/micha/Desktop/School/COMP3106/COMP3106A2/Examples/Example0/dataset.csv", "C:/Users/micha/Desktop/School/COMP3106/COMP3106A2/Examples/Example0/patient_measurements.txt")
  #print most likely class and probabilities
  print(most_likely_class, class_probabilities)
  


main()
