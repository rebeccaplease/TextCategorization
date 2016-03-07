from __future__ import print_function, division

import nltk as nltk
from nltk.tokenize import word_tokenize
from numpy import empty
import os

from datetime import datetime

class Document:
    def __init__(self, filepath, category):
        self.tf = dict()
        self.file = filepath
        self.cat = category
    def addTF(self, word):
        if word in self.tf:
            self.tf[word] += 1
            return False
        else:
            self.tf[word] = 1 #first occurence of word in this document
            return True

    def __str__(self):
        return str(self.tf) + self.cat
    def __repr__(self):
        return str(self.tf) + self.cat

     #str(self.tf) + self.cat

#prompt for user input
print("\n---Text Categorization Training Program---\n")

start = datetime.now()

## Read in filename and open file
valid = False
while not valid:
    try:
        #trainingFile = raw_input("Enter training document file name: ")
        #f = open(trainingFile) # read in training
        f = open('./TC_provided/corpus1_train.labels', 'r')
        valid = True
        # format: relativePath/filename category
    except IOError:
        print("Please enter a valid filename")
#outputFile = raw_input("Enter trained output filename: ")

## Read file: find categories and read in filenames
files = list() #create an empty list
categories = set() #create an empty set
N = 0 #keep track of total documents

for line in f:
    #print(line)
    temp = line.split() # temp(0) has file location, temp(1) has categories
    if len(temp) > 2:
        print("File error: Invalid format.")
    files.append(Document(temp[0], temp[1]))
    categories.add(temp[1]) # add to set
    N += 1

#print(files)
#print(categories)

f.close() #close file after done

## train on training set for each document in files

# loop through and find tf for each doc
# find idf for each term - set true for each encountered term in a doc

categories = list(categories) #convert set to list

#catDoc = empty(len(categories)) #empty array to be filled with lists
catDoc = list() #empty list
for c in range(len(categories)): #fill array with lists to hold docs
    catDoc.append(list()) #list of lists

allWords = dict()

while(len(files) > 0):
    doc = files.pop()
    # parse file
    # open file at filepath
    # loop through words

    # lol how do relative path
    #directory = os.path.dirname(os.path.abspath(__file__))
    #print(directory)
    #relpath = '/TC_provided'+doc.file[1:]
    #print('relpath: ', relpath)
    #filename = os.path.join(directory, relpath) #don't include .
    #print(filename)
    #f = open(filename)


    #directory = os.path.dirname(os.path.abspath(__file__))
    #relativePath = os.path.relpath(doc.file) #get relative path of the document
    #filename = os.path.join(directory, relativePath)
    #print("directory: ",directory)
    #print("relative path: ",relativePath)
    #print("file name: ", filename)
    #f = open(filename)
     # combine with absolute current directory

    relativePath = os.path.relpath(doc.file) #get relative path of the document
    filename = os.path.join("TC_provided", relativePath)
    #print("relative path: ",relativePath)
    #print("file name: ", filename)
    f = open(filename)

    for line in f:
        words = word_tokenize(line) # tokenized words in array
        for w in words:
            #check for punctuations, contractions
            if doc.addTF(w): # first time word has occured in doc, add idf score
                if w in allWords: #if the word already is there
                    allWords[w] += 1
                else:
                    allWords[w] = 1
    f.close()
    catDoc[categories.index(doc.cat)].append(doc) #add document to correct category with its term weights

f = open('output.txt','w+')
f.write(str(catDoc))
f.close()
#print(catDoc[0])

end = datetime.now()
print("\nRuntime: ",end-start)
#print(catDoc)
# iterate through catDoc and calculate tf*idf
# take average for each term to get the centroid for each cat

#for docList in catDoc:




## print out trained results: categories and feature vectors, weights, etc
# print out catDoc
