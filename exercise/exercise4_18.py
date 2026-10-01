#Exercise 4.18
#Modify the Python program from Example 4.22, analyze a different piece of text.

import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = """
Machine learning (ML) is a type of artificial intelligence (AI) that allows 
software applications to become more accurate at predicting outcomes without 
being explicitly programmed to do so. Machine learning algorithms use 
historical data as input to predict new output values.
Andrew Ng is one of the world's most famous machine learning researchers. 
He co-founded Coursera and the Google Brain project.
"""

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)
