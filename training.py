from __future__ import print_function, division

import nltk as nltk
from nltk.tokenize import word_tokenize
import os

from datetime import datetime

from numpy import zeros

from math import log10, sqrt

class Document:
    def __init__(self, filepath, category):
        self.tf = dict()
        self.file = filepath
        self.cat = category
        self.predicted = "" #predicted category by testing
        self.wordCount = 0
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


def readTrainingFiles(files, categories):
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
    #files = list() #create an empty list
    #categories = set() #create an empty set
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

    f.close() #close file after done reading filepaths and categories
    return N

## train on training set for each document in files
# iterate through catDoc and calculate tf*idf
# take average for each term to get the centroid for each cat

def getTFIDF(files, allWords, N, categories):
    catDoc = list() #empty list
    for c in range(len(categories)): #fill array with lists to hold docs
        catDoc.append(list()) #list of lists

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
        numWordsInDoc = 0
        for line in f:
            words = word_tokenize(line) # tokenized words in array

            for w in words:
                #check for punctuations, contractions
                if doc.addTF(w): # first time word has occured in doc, add doc count score
                    if w in allWords: #if the word already is there
                        allWords[w] += 1
                    else:
                        allWords[w] = 1
                    numWordsInDoc += 1 # increment number of words for this document
        # ( normalize tf by total number of terms?)
        f.close()
        doc.wordCount = numWordsInDoc
        catDoc[categories.index(doc.cat)].append(doc) #add document to correct category with its term weights

    # calculate idf score - loop through allWords and d
    idf = dict()
    for word, count in allWords.items():
        idf[word] = log10(N/count)

    weights = list() #empty list - hold centroid of each category
    for c in range(len(categories)): #fill array with lists to hold docs
        weights.append(dict()) # dictionary maps word: weight

    # calculate tf*idf for each category, keep running avg for each
    for k in range(len(categories)): # loop through catDoc
        docs = catDoc[k] # list of documents in a category
        docCount = len(docs)
        for d in docs: # iterate through docs in this category and calculate tf*idf score for each word in document
            #keep running square of tf*idf
            docNorm = 0
            docVector = dict()
            for word, tfcount in d.tf.items(): # iterate through words in document
                idfWord = idf.get(word) # get idf for this word
                calcWeight = ((tfcount/d.wordCount)*idfWord)/docCount
                if word in docVector:
                    docVector[word] +=  calcWeight # calculate weight, word:weight. normalize tf by wordcount
                    # and normalize overall weight by the document count
                else:
                    docVector[word] = calcWeight # calculate weight, word:weight. normalize tf by wordcount
                    # and normalize overall weight by the document count
                docNorm += pow(calcWeight,2) #square tf*idf score for this word
            docNorm = sqrt(docNorm) #take square root
            # normalize each term of the document vector by calculated norm and the number of docs in this cat
            for word, weight in docVector.items():
                normalizedDocWeight = weight/docNorm/docCount
                docVector[word] = normalizedDocWeight
                # add to weights vector
                if word in weights[k]:
                    weights[k][word] += normalizedDocWeight
                else:
                    weights[k][word] = normalizedDocWeight
    return weights

def outputTrainingFile(categories, weights):
    f = open('output.txt','w+')
    #print out number of categories
    for k in range(len(categories)):
        # write out list of all words and idf score
        f.write("-"+categories[k]+"-"+ "\n\n"+str(weights[k])+"\n\n")
        #write out number of words in each category - read into testing.py
    f.close()


## testing here (figure out file format after ##

## Read test doc filenames

def readTestFiles(testFilesList, categories, weights, allWords):
    valid = False
    while not valid:
        try:
            #testFile = raw_input("Enter test document file name: ")
            testFile = './TC_provided/corpus1_test.labels'
            f = open(testFile, 'r') # read in trained file
            valid = True
            # format: relativePath/filename category
        except IOError:
            print("Please enter a valid filename")

    #outputFile = raw_input("Enter output filename: ")

    for line in f:
        temp = line.split() # filename, category
        testFilesList.append(Document(temp[0], temp[1])) # filename to parse

    f.close() #close file after done

## test


# compare to trained category weights

#iterate through words in test document,
#compute similarity -
#if centroid category is already normalized, no need to normalize the test document
#because the length will be the same anyways
def assignCat(testFilesList, categories, weights, allWords):
    for k in range(len(testFilesList)):
        #find tf for each word in each test document
        doc = testFilesList[k]
        relativePath = os.path.relpath(doc.file) #get relative path of the document
        filename = os.path.join("TC_provided", relativePath)

        f = open(filename, 'r')
        numWordsInDoc = 0
        for line in f:
            words = word_tokenize(line) # tokenized words in array
            for w in words:
                #check for punctuations, contractions
                doc.addTF(w) # first time word has occured in doc
                numWordsInDoc += 1 # increment number of words for this document
        f.close()
        doc.wordCount = numWordsInDoc

    #calculate similarity - multiply tf of each test doc by the centroid of each cat

    for doc in testFilesList:
        maxSim = 0
        catIndex = -1
        for k in range(len(weights)): # iterate through each category
            sim = 0
            #iterate through each word in test doc
            for word, count in doc.tf.items():
                sim += weights[k].get(word,0)*allWords.get(word, 0)*count
            if sim > maxSim: #store max similarity of all categories and index
                maxSim = sim
                catIndex = k
        #print(catIndex)
        doc.predicted = categories[catIndex] #assign category to current document

def calculateMetrics(testFilesList, categories):
    ## print out results: filename category

    total = 0 #keep track of total number of documents (or from testing)
    confusion = zeros((len(categories), len(categories))) # x corresponds to correct cat, y corresponds to predicted

    # compare to test file
    for k in range(len(testFilesList)):
        catCorrect = testFilesList[k].cat
        catPredicted = testFilesList[k].predicted # get predicted cat from test
        x = categories.index(catCorrect) # get index categories to put into matrix
        y = categories.index(catPredicted)
        confusion[x][y] += 1 # increment count by 1
        total += 1

    A = 0 # predict cat, expect cat (true positive)
    B = 0 # predict cat, expect no (false positive)
    C = 0 # predict no, expect cat (false negative)
    D = 0 # predict no, expect no (true negative)

    # microaverage: for each cat, take average
    # macroaverage: overall
    # https://en.wikipedia.org/wiki/Confusion_matrix

    metrics = zeros((len(categories), 4)) # hold metrics (ABCD) for each category
    # 0 = A, 1 = B, 2 = C, 3 = D
    A = 0
    B = 0
    C = 0
    D = 0
    # loop through confusion matrix
    for i in range(len(categories)): # for each predicted category
        for j in range(len(categories)): #for each actual category
            num = confusion[i][j]
            if i == j: # prediction matches actual cat
                metrics[i][0] += num
                A += num
            else:
                metrics[i][1] += num #if prediction falsely says cat, false positive
                metrics[j][2] += num #if prediction falsely says not cat, false negative
                B += num
                C += num

    microOverallAcc = 0
    microPrecision = 0
    microRecall = 0
    microF1 = 0

    for k in range(len(categories)): # for each predicted category
        # true negative: difference between a,b,c,d metrics for this cat and the total number of docs
        metrics[k][3] = total - (metrics[k][0] + metrics[k][1] + metrics[k][2])
        #microaverage: for each cat, then take average
        a = metrics[k][0]
        b = metrics[k][1]
        c = metrics[k][2]
        d = metrics[k][3]
        microOverallAcc += (a+d)/(a+b+c+d)/len(categories)
        microPrecision += a/(a+b)/len(categories)
        microRecall += a/(a+c)/len(categories)
        microF1 += (2*microPrecision*microRecall)/(microPrecision+microRecall)/len(categories)


    #macroaverage
    overallAccuracy = (A+D)/(A+B+C+D)
    precision = A/(A+B)
    recall = A/(A+C)
    f1 = (2*precision*recall)/(precision+recall)


    #print results
    #microaverage
    print("-Microaverage-")
    print("Overall accuracy: ",microOverallAcc)
    print("Precision: ",microPrecision)
    print("Recall: ",microRecall)
    print("F1: ",microF1)

    #macroaverage
    print("-Macroaverage-")
    print("Overall accuracy: ",overallAccuracy)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("F1: ",f1)

def outputResults(testFilesList):
    f = open('outputTesting.txt','w+')
    #print out number of categories
    for k in range(len(testFilesList)):
        # write out list of all words and idf score
        #f.write(testFilesList[k].file+ " "+ testFilesList[k].predicted+ " "+ testFilesList[k].cat+"\n")
        f.write(testFilesList[k].file+ " "+ testFilesList[k].predicted+"\n")
        #write out number of words in each category - read into testing.py
    f.close()

if __name__ == "__main__":
    print("\n---Text Categorization Training Program---\n")

    start = datetime.now()
    files = list() #create an empty list
    categories = set() #create an empty set

    numDocs = readTrainingFiles(files, categories)
    categories = list(categories) #convert set to list

    allWords = dict() # hold all words for idf calculation - inverted index
    weights = getTFIDF(files, allWords, numDocs, categories)
    outputTrainingFile(categories, weights)

    end = datetime.now()
    print("\nRuntime: ",end-start)

    ##testing##
    print("\n---Text Categorization Testing Program---\n")

    start = datetime.now()
    testFilesList = list() # create an empty list for holding test files
    #empty list for categories
    readTestFiles(testFilesList, categories, weights, allWords)
    assignCat(testFilesList, categories, weights, allWords)

    outputResults(testFilesList)

    end = datetime.now()
    print("\nRuntime: ",end-start)
