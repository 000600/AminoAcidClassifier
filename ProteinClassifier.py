# Imports
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from random import randint

## Bases
# A - 1
# T - 2
# G - 3
# C - 4
# U - 5

transcription_letter_map = {1 : "A", 2 : "T", 3 : "G", 4 : "C", 5 : "U"} # Turning bases into numbers for the neural network
transcription_map = {2 : 1, 1 : 5, 3 : 4, 4 : 3} # Actual transcription process

# Codons to proteins
translation_map = {
       "UUU":"Phenylalanine", "UUC":" Phenylalanine", 
       "UUA":"Leucine", "UUG":"Leucine",
       "UCU":"Serine", "UCC":"Serine", "UCA":"Serine", "UCG":"Serine",
       "UAU":"Tyrosine", "UAC":"Tyrosine", "UAA":"STOP", "UAG":"STOP",
       "UGU":"Cysteine", "UGC":"Cysteine", "UGA":"STOP", "UGG":"Tryptophan",
       "CUU":"Leucine", "CUC":"Leucine", "CUA":"Leucine", "CUG":"Leucine",
       "CCU":"Proline", "CCC":"Proline", "CCA":"Proline", "CCG":"Proline",
       "CAU":"Histidine", "CAC":"Histidine", "CAA":"Glutamine", "CAG":"Glutamine",
       "CGU":"Arginine", "CGC":"Arginine", "CGA":"Arginine", "CGG":"Arginine",
       "AUU":"Isoleucine", "AUC":"Isoleucine", "AUA":"Isoleucine", "AUG":"Methionine",
       "ACU":"Threonine", "ACC":"Threonine", "ACA":"Threonine", "ACG":"Threonine",
       "AAU":"Asparagine", "AAC":"Asparagine", "AAA":"Lysine", "AAG":"Lysine",
       "AGU":"Serine", "AGC":"Serine", "AGA":"Serine", "AGG":"Serine",
       "GUU":"Valine", "GUC":"Valine", "GUA":"Valine", "GUG":"Valine",
       "GCU":"Alanine", "GCC":"Alanine", "GCA":"Alanine", "GCG":"Alanine",
       "GAU":"Aspartic acid", "GAC":"Aspartic acid", "GAA":"Glutamic acid", "GAG":"Glutamic acid",
       "GGU":"Glycine", "GGC":"Glycine", "GGA":"Glycine", "GGG":"Glycine"
       }

# Get classes
classes = ['Methionine', 'Histidine', 'Threonine', 'Valine', 'Leucine', 'Proline', 'Phenylalanine', 'Aspartic acid', 'Glutamine', 'Arginine', 'Tyrosine', ' Phenylalanine', 'Glutamic acid', 'Tryptophan', 'STOP', 'Asparagine', 'Cysteine', 'Isoleucine', 'Lysine', 'Glycine', 'Serine', 'Alanine']
encoded_classes = {}

# Encode classes for the network
for index in range(len(classes)):
    encoded_classes[classes[index]] = index

# Get reversed classes for prediction purposes
reversed_classes = {v: k for k, v in encoded_classes.items()}

# Generate data
def generate_dataset(num_sets): # More codons --> more data
    # Generate codons
    x = []

    for i in range(num_sets):
      reverse1 = {v: k for k, v in transcription_letter_map.items()}
      reverse2 = {v: k for k, v in transcription_map.items()}
      
      # Get all unique codons so that the dataset is representative of all possible inputs
      for string in translation_map:
          codon = []
          for letter in string:
              codon.append(reverse2[reverse1[letter]])
          x.append(codon) # Each set contains approximately 64 codons
    return x

# Generate labels
def transcription_and_translation(x):
    transcribed = []
    translated = []
    # Transcribe codons
    for codon in x:
        codon_holder = []
        for base in codon:
            new_base = transcription_map[base]
            codon_holder.append(new_base)
        transcribed.append(codon_holder)

    # Translate Codons
    for codon in transcribed:
        string = ""
        for base in codon:
            string += transcription_letter_map[base] # Add all bases together to get a string version of the codon

        translated.append(encoded_classes[translation_map[string]])
    return translated

# Turn all inputs into codons with letter bases for interpretation purposes (unencode the dataset)
def inputs_to_codons(input):
    transcribed_codons = []
    proteins = []
    for codon in input:
        string = ""
        for base in codon:
          string += transcription_letter_map[base]

        transcribed_codons.append(string)
    return transcribed_codons

# Create datasets
trainx = generate_dataset(100) # Training set 
trainy = transcription_and_translation(trainx)
testx = generate_dataset(30) # Testing set (for evaluation)
testy = transcription_and_translation(testx)

# View data size
print("Training Dataset Size:", len(trainx))
print("Testing Dataset Size:", len(testx))

# Create model
model = Sequential()

# Input layer
model.add(Dense(3, input_shape = [3])) # input_shape must be 3 since the network is receiveing codons which have 3 bases

# Hidden layers
model.add(Dense(258, activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(258, activation = "relu"))
model.add(Dense(126, activation = "relu"))
model.add(Dense(258, activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(1024, activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(258, activation = "relu"))
model.add(Dense(126, activation = "relu"))

# Output layer
model.add(Dense(22)) # 22 output neurons because there are 22 individual proteins (and each protein is a class)

# Optimizer
opt = Adam(learning_rate = 0.001)

# Compile and train model
model.compile(optimizer='adam', loss = SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])
model.fit(trainx, trainy, epochs = 10)

# Evaluate model
test_loss, test_acc = model.evaluate(np.array(testx), np.array(testy), verbose = 2)

# View inputs
input_codons = inputs_to_codons(trainx)
print("\n\nTrain Inputs (Codons):", input_codons)

# View corresponding outputs
train_labels = []
for i in trainy:
  train_labels.append(reversed_classes[i])
print("\n\nTrain Labels (Proteins):", train_labels)

# View test accuracy
print('\nTest Aacuracy:', test_acc)
print("**************")

# Predict
print("\n\nSAMPLE PREDICTION")
print("=================")

codon = [[2, 1, 4]] # Get codon
label = transcription_and_translation(codon) # Get label
print("\nCodon:", codon)

prediction = model.predict(codon)
print("Prediction Probabilities:")
print(prediction) # Get array of model probability predictions
prediction_index = np.argmax(prediction) # Get model's prediction (the class the model calculates to have the highest probability of being the label)

# View model's prediction compared to the actual values
print("\nModel's Prediction:", reversed_classes[prediction_index], "| Actual Label:", reversed_classes[label[0]])
