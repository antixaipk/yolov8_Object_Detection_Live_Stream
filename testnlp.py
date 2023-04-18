import spacy

# Load the pre-trained model
nlp = spacy.load('en_core_web_lg')

# Define the two texts
text1 = "This is the first text about machine learning. It discusses the different types of algorithms used in machine learning and their applications."
text2 = "This is the second text about artificial intelligence. It covers the history of AI, the current state of the technology, and its potential future impact."

# Process the texts using the model
doc1 = nlp(text1)
doc2 = nlp(text2)

# Compute the semantic similarity
similarity = doc1.similarity(doc2)

print(similarity) # Print the similarity score
