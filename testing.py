from __future__ import print_function, division

import nltk
import numpy

#prompt for user input
print("---Text Categorization Testing Program---")

## Read in filename and open file
valid = False
while not valid:
    try:
        trainedFile = raw_input("Enter trained document file name: ")
        f = open(trainedFile) # read in trained file
        valid = True
        # format: relativePath/filename category
    except IOError:
        print("Please enter a valid filename")

## Read file: trained format: categories and features
files = []] # create an empty array
categories = f.nextline.split()
#read in weights, features, etc

f.close()


valid = False
while not valid:
    try:
        testFile = raw_input("Enter test document file name: ")
        f = open(testFile) # read in trained file
        valid = True
        # format: relativePath/filename category
    except IOError:
        print("Please enter a valid filename")

#outputFile = raw_input("Enter output filename: ")

for line in f:
    files.append(Document(line)) # filename to parse

print(files)
print(categories)

f.close() #close file after done

## test

## print out results: filename category

## if corpus1, compare to test file
# cross validation
try:
    f = open(testFile) # read in trained file
    valid = True
    # format: relativePath/filename category
except IOError:
    print("Test file not found")

iterator = 0
total = 0 #keep track of total number of documents (or from testing)
confusion = zeros(len(category), len(category)) # x corresponds to correct cat, y corresponds to predicted

# compare to test file
for line in f:
    temp = line.split() #temp[0] holds the filename, temp[1] holds the correct category
    catCorrect = temp[1] # get correct cat from test file
    catPredicted = files[iterator++].cat # get predicted cat from test
    x = categories.index(catCorrect) # get index categories to put into matrix
    y = categories.index(catPredicted)
    confusion[x][y]++ # increment count by 1
    total++

f.close()

A = 0 # predict cat, expect cat (true positive)
B = 0 # predict cat, expect no (false positive)
C = 0 # predict no, expect cat (false negative)
D = 0 # predict no, expect no (true negative)

# microaverage: for each cat, take average
# macroaverage: overall
# https://en.wikipedia.org/wiki/Confusion_matrix

metrics = zeros(len(category), 4) # hold metrics (ABCD) for each category
# 0 = A, 1 = B, 2 = C, 3 = D
A = 0
B = 0
C = 0
D = 0
# loop through confusion matrix
for i in range(len(category)): # for each predicted category
    for j in range(len(category)): #for each actual category
        num = category[i][j]
        if i == j # prediction matches actual cat
            metrics[i][0] += num
            A += num
        else
            metrics[i][1] += num #if prediction falsely says cat, false positive
            metrics[j][2] += num #if prediction falsely says not cat, false negative
            B += num
            C += num
for k in range(len(category)): # for each predicted category
    # true negative: difference between a,b,c,d metrics for this cat and the total number of docs
    metrics[k][3] = total - (metrics[k][0] + metrics[k][1] + metrics[k][2])

#macroaverage
overallAccuracy = (A+D)/(A+B+C+D)
precision = A/(A+B)
recall = A/(A+C)
f1 = (2*precision*recall)/(precision+recall)

#microaverage: for each cat, then take average 

class Document:
    def __init__(self, filepath, category):
        self.file = filepath
        self.cat = category
    def __init__(self, filepath):
        self.file = filepath
    def addCategory(self, category):
        self.cat = category
