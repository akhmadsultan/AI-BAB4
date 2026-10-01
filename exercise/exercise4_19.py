#Exercise 4.19
#Modify the Python program from Example 4.23, test it with different sentences (positive, negative, neutral).

from transformers import pipeline
classifier = pipeline('sentiment-analysis')

sentences = [
 'This is a good movie.',
 'This is a bad movie.',
 'The weather is okay today.'
]

for sentence in sentences:
 result = classifier(sentence)
 print(sentence, '->', result)
