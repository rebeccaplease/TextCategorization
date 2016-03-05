from __future__ import print_function, division

import nltk

#prompt for user input
print("---Text Categorization Training Program---")

## Read in filename and open file
valid = False
while not valid:
    try:
        trainingFile = raw_input("Enter training document file name: ")
        f = open(trainingFile) # read in training
        #f = open('./TC_provided/corpus1_test.labels')
        valid = True
        # format: relativePath/filename category
    except IOError:
        print("Please enter a valid filename")
#outputFile = raw_input("Enter trained output filename: ")

## Read file: find categories and read in filenames
files = list() #create an empty list
categories = set() #create an empty set

for line in f:
    #print(line)
    temp = line.split() # temp(0) has file location, temp(1) has categories
    if len(temp) > 2:
        print("File error: Invalid format.")
    files.append(temp[0])
    categories.add(temp[1]) # add to set

print(files)
print(categories)

f.close() #close file after done

## train on training set for each document in files

## print out trained results: categories and feature vectors, weights, etc
