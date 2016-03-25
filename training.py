from __future__ import print_function, division

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import re

import os

from datetime import datetime


from math import log10, sqrt

from nltk.tag.perceptron import PerceptronTagger
#globals#
tagger = PerceptronTagger()
wnl = WordNetLemmatizer()
punct = ['/','.',',','!','?','-',';',':']
punct = set(punct)

#stopWords = ['a','the','an', "n't", 'not', "'s",'is', 'which','on']
#stopWords = set(stopWords)
#regular expressions
#date to date represntaton
#numbers to number represenation
#lowercase?
#contractions
#possessive  -ignore 's is
NORM = 1

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
            filepath = os.path.relpath(filename)
            f = open(filepath)
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
    f.close()
    return N

# Translate treebank POS tag to wordnet POS tag
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
    filename = os.path.relpath(doc.file)
    f = open(filename)
    numWordsInDoc = 0
    for line in f:
        #line = line.lower()
        #line = re.sub('(Jan|January|Feb|February|March|Mar|April|Apr|May|Jun|June\
        #    July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|\
        #    Dec)
        #    ,'dddd',line)
        words = word_tokenize(line)
        tagset = None
        words = nltk.tag._pos_tag(words, tagset, tagger) #[('And', 'CC'), ...]
        for k in range(len(words)):
            partOfSpeech = get_wordnet_pos(words[k][1])
            if partOfSpeech == '':
                words[k] = wnl.lemmatize(words[k][0])
            else:
                words[k] = wnl.lemmatize(words[k][0], pos=partOfSpeech)
        for w in words:
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
    f.close()
    doc.wordCount = numWordsInDoc

# Returns matrix of documents and TF*IDF weight
def getTFIDF(files, allWords, N, categories):
    catDoc = list()
    numCat = len(categories)
    for c in range(numCat):
        catDoc.append(list())

    for doc in files:
        calculateTF(doc, allWords, True)
        catDoc[categories.index(doc.cat)].append(doc) #add document to correct category with its term weights

    for word in allWords:
        allWords[word] = log10(N/allWords[word])

    weights = list() # hold centroid of each category
    for k in range(numCat):
        weights.append(dict())
        docs = catDoc[k]
        docCount = len(docs)
        for d in docs:
            docNorm = 0
            docVector = dict()
            for word, tfcount in d.tf.items():
                idfWord = allWords.get(word)
                calcWeight = (tfcount/d.wordCount)*idfWord/docCount
                if word in docVector:
                    docVector[word] +=  calcWeight
                else:
                    docVector[word] = calcWeight
                docNorm += pow(calcWeight,2)
            docNorm = sqrt(docNorm)
            # normalize each term of the document vector by calculated norm and the number of docs in this cat
            for word, weight in docVector.items():
                normalizedDocWeight = weight/docNorm/docCount
                docVector[word] = normalizedDocWeight
                if word in weights[k]:
                    weights[k][word] += normalizedDocWeight
                else:
                    weights[k][word] = normalizedDocWeight
    return weights

def readTestFiles(testFiles, categories, weights, allWords):
    f = openFile("Enter test document file name: ")
    for line in f:
        testFiles.append(Document(line.split()[0], ""))
    f.close()

#calculate similarity and assign best
def assignCat(testFiles, categories, weights, allWords):
    for doc in testFiles:
        calculateTF(doc, allWords, False)
        maxSim = 0
        catIndex = -1
        for k in range(len(weights)): # iterate through each category
            sim = 0
            for word, count in doc.tf.items():
                sim += (NORM+weights[k].get(word,0))*(count/doc.wordCount)*(NORM+allWords.get(word, 0))
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

    trainingFiles = list()
    categories = set()
    numDocs = readTrainingFile(trainingFiles, categories)
    start = datetime.now()
    categories = list(categories)

    allWords = dict() #idf score
    TFIDFweights = getTFIDF(trainingFiles, allWords, numDocs, categories)

    end = datetime.now()
    print("\nRuntime: ",end-start)

    print("\n---Text Categorization Testing Program---\n")

    testFiles = list()

    readTestFiles(testFiles, categories, TFIDFweights, allWords)
    start = datetime.now()
    assignCat(testFiles, categories, TFIDFweights, allWords)

    end = datetime.now()
    print("\nRuntime: ",end-start)

    outputResults(testFiles)

if __name__ == "__main__":
    main()
