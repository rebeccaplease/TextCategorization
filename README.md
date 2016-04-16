# TextCategorization

ECE 467 Natural Language Processing
 
Spring 2015 

http://faculty.cooper.edu/sable2/courses/spring2016/ece467/

The training and testing program is located in one file: rocchio.py and it can be run using 
```
python roccio.py 
```
in the command line.  I used the Rocchio TF\*IDF method.  Tokenization was performed using  word_tokenize from nltk.  The TF score was normalized by the total number of tokens in the document + 1 for tokens that did not appear in the training set.  The IDF score for each word was calculated using log N/Ni where N is the total number of documents and Ni is the number of documents the token occurs in.  The TF*IDF centroid vector for each category was normalized by the root mean square of the weights of each category. The cosine similarity metric was used to compare each document to the centroid vectors - the largest result was assigned as the cateogory. 
