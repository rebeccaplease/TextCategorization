from __future__ import print_function, division

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import os

from datetime import datetime

from numpy import zeros

from math import log10, sqrt

from nltk.tag.perceptron import PerceptronTagger
#globals#
tagger = PerceptronTagger()
wnl = WordNetLemmatizer()
punct = ['/','.',',','!','?','-',';',':']
#punct = ['/','.',',','!','?','-',';',':','+','@','#']
punct = set(punct)

stopWords = ['a','the','an','']
stopWords = set(stopWords)
NORM = 0.07

class Document:
    def __init__(self, filepath, category):
        self.tf = dict()
        self.file = filepath
        self.cat = category
        self.predicted = ""
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

def openFile(prompt):
    valid = False
    while not valid:
        try:
            filename = raw_input(prompt)
            #relativePath = os.path.relpath(filename) #get relative path of the document
            #filepath = os.path.join("TC_provided", relativePath)
            #filepath = os.path.join(os.path.dirname(__file__), filename)
            #print(os.path.curdir)
            filepath = os.path.relpath(filename)
            #print(filepath)
            f = open(filepath) # read in training
            valid = True
        except IOError:
            print("Please enter a valid filename")
    return f

def readTrainingFile(files, categories):
    f = openFile("Enter training document file name: ")
    N = 0
    for line in f:
        temp = line.split()
        if len(temp) > 2:
            print("File error: Invalid format.")
        files.append(Document(temp[0], temp[1])) #filepath, cat
        categories.add(temp[1])
        N += 1
    f.close() #close file after done reading filepaths and categories
    return N

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
# Update TF for this doc and allWords
def calculateTF(doc, allWords, training):
    #print(doc.file)
    filename = os.path.relpath(doc.file) #get relative path of the document
    #filename = os.path.join("TC_provided", relativePath)
    f = open(filename)
    #filepath = os.path.join(os.path.dirname(__file__), doc.file)

    numWordsInDoc = 0
    for line in f:
        #print("send help")
        #q = datetime.now()
        words = word_tokenize(line) # tokenized words in array
        #print("tokenize ", datetime.now()-q)
        tagset = None
        q = datetime.now()
        words = nltk.tag._pos_tag(words, tagset, tagger) #[('And', 'CC'), ...]
        #print("pos_tag ", datetime.now()-q)
        #q = datetime.now()
        for k in range(len(words)):
            partOfSpeech = get_wordnet_pos(words[k][1])
            if partOfSpeech == '':
                words[k] = wnl.lemmatize(words[k][0])
            else:
                words[k] = wnl.lemmatize(words[k][0], pos=partOfSpeech)
        #print("lemm ", datetime.now()-q)

        #q = datetime.now()
        #print(words)
        for w in words:
            #check for punctuations, contractions
            #combine = '/'.join(w)
            #print(combine)

            if w not in punct:
                if training:
                    if doc.addTF(w):
                        if w in allWords:
                            allWords[w] += 1
                        else:
                            allWords[w] = 1
                else:
                    doc.addTF(w)
                numWordsInDoc += 1
        #print("add to allwords ", datetime.now()-q)
    f.close()

    doc.wordCount = numWordsInDoc

# Returns matrix of documents and TF*IDF weight
def getTFIDF(files, allWords, N, categories):
    catDoc = list() #empty list
    for c in range(len(categories)): #fill array with lists to hold docs
        catDoc.append(list()) #list of lists

    for doc in files:
        calculateTF(doc, allWords, True)
        catDoc[categories.index(doc.cat)].append(doc) #add document to correct category with its term weights

    #idf = dict()
    #for word, count in allWords.items():
        #idf[word] = log10(N/count)
    # calculate idf score
    for word in allWords:
        allWords[word] = log10(N/allWords[word])
    #print(allWords)

    weights = list() # hold centroid of each category

    # calculate tf*idf for each category, keep running avg for each
    for k in range(len(categories)): # loop through catDoc
        weights.append(dict()) # dictionary maps word: weight
        docs = catDoc[k]
        docCount = len(docs)
        #print(k)
        for d in docs: # iterate through docs in this category and calculate tf*idf score for each word in document
            #keep running square of tf*idf
            docNorm = 0
            docVector = dict()
            for word, tfcount in d.tf.items(): # iterate through words in document
                idfWord = allWords.get(word) # get idf for this word

                calcWeight = ((tfcount/d.wordCount)*idfWord)/docCount
                if word in docVector:
                    docVector[word] +=  calcWeight # calculate weight, word:weight. normalize tf by wordcount
                    # and normalize overall weight by the document count
                else:
                    docVector[word] = calcWeight
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
    print(weights)
    return weights

def outputTrainingFile(categories, weights):
    f = open('output.txt','w+')
    #print out number of categories
    for k in range(len(categories)):
        # write out list of all words and idf score
        f.write("-"+categories[k]+"-"+ "\n\n"+str(weights[k])+"\n\n")
        #write out number of words in each category - read into testing.py
    f.close()

def readTestFiles(testFiles, categories, weights, allWords):
    f = openFile("Enter test document file name: ")
    for line in f:
        testFiles.append(Document(line.split()[0], ""))
    f.close()

#calculate similarity and assign best
def assignCat(testFiles, categories, weights, allWords):
    for doc in testFiles:
        calculateTF(doc, allWords, False)

    for doc in testFiles:
        maxSim = 0
        catIndex = -1
        for k in range(len(weights)): # iterate through each category
            sim = 0
            #iterate through each word in test doc
            for word, count in doc.tf.items():
                #score = weights[k].get(word,-1)*(count/doc.wordCount)*allWords.get(word, 1)
                #if score < 0:
                    #sim += NORM
                #else:
                sim += weights[k].get(word,0)*(count/doc.wordCount)*allWords.get(word, 0)
            if sim > maxSim: #store max similarity of all categories and index
                maxSim = sim
                catIndex = k
        doc.predicted = categories[catIndex] #assign category to current document

def outputResults(testFilesList):
    filename = raw_input("Please enter an output filename: ")

    f = open(filename,'w+')
    for k in range(len(testFilesList)):
        f.write(testFilesList[k].file+ " "+ testFilesList[k].predicted+"\n")
    f.close()

def main():
    print("\n---Text Categorization Training Program---\n")
    start = datetime.now()

    trainingFiles = list()
    categories = set()

    numDocs = readTrainingFile(trainingFiles, categories)
    categories = list(categories) #convert set to list

    allWords = dict() # hold all words for idf calculation - inverted index
    TFIDFweights = getTFIDF(trainingFiles, allWords, numDocs, categories)
    #outputTrainingFile(categories, TFIDFweights)

    end = datetime.now()
    print("\nRuntime: ",end-start)

    print("\n---Text Categorization Testing Program---\n")

    start = datetime.now()
    testFiles = list() # create an empty list for holding test files

    readTestFiles(testFiles, categories, TFIDFweights, allWords)
    assignCat(testFiles, categories, TFIDFweights, allWords)

    end = datetime.now()
    print("\nRuntime: ",end-start)

    outputResults(testFiles)

if __name__ == "__main__":
    main()
