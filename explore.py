import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from collections import defaultdict
import time
import os

# Set the path to the manually downloaded NLTK data
nltk_data_path = '/Users/satrio/nltk_data'
nltk.data.path.append(nltk_data_path)

# Ensure you have downloaded the necessary nltk data files
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Necessary NLTK data files not found. Please check the nltk_data_path.")

# Load the dataset
file_path = 'News.csv'
df = pd.read_csv(file_path)

# Fill NaN values in the 'content' column with an empty string
df['content'] = df['content'].fillna('')

# Extract the 'content' column
contents = df['content']

# Number of documents
num_docs = contents.shape[0]

# Initialize counters
num_words = 0
num_sentences = 0

# Process each document
for content in contents:
    # Count words
    words = content.split()
    num_words += len(words)
    
    # Count sentences
    sentences = nltk.sent_tokenize(content)
    num_sentences += len(sentences)

print(f"Number of documents: {num_docs}")
print(f"Total number of words: {num_words}")
print(f"Total number of sentences: {num_sentences}")

# List of Indonesian stopwords
indonesian_stopwords = set(stopwords.words('indonesian'))

# Additional common stopwords (add any that are missing)
additional_stopwords = {"seperti", "oleh", "untuk", "dari", "pada", "dengan", "yang", "atau", "dan"}
indonesian_stopwords.update(additional_stopwords)

# Function to remove stopwords and create terms
def preprocess_text(text):
    terms = re.findall(r'\w+', text.lower())
    filtered_terms = [term for term in terms if term not in indonesian_stopwords]
    return filtered_terms

# Initialize the inverted index
inverted_index = defaultdict(list)

# Build the inverted index
for doc_id, content in enumerate(contents):
    terms = preprocess_text(content)
    for term in set(terms):  # Use set to avoid duplicate entries in the same document
        inverted_index[term].append(doc_id)

# Display a sample of the inverted index
# for term, doc_ids in list(inverted_index.items())[:10]:
#     print(term, doc_ids)

def boolean_retrieval_and(term1, term2):
    return set(inverted_index[term1]).intersection(set(inverted_index[term2]))

def boolean_retrieval_or(term1, term2):
    return set(inverted_index[term1]).union(set(inverted_index[term2]))

def boolean_retrieval_not(term):
    return set(range(num_docs)) - set(inverted_index[term])

# AND Queries
start_time = time.time()
and_result = boolean_retrieval_and('covid', 'vaksin')
end_time = time.time()
print(f"AND query result for covid and vaksin: {and_result} (Time taken: {end_time - start_time}s)")

start_time = time.time()
and_result = boolean_retrieval_and('vaksin', 'indonesia')
end_time = time.time()
print(f"AND query result for vaksin and indonesia: {and_result} (Time taken: {end_time - start_time}s)")

# OR Queries
start_time = time.time()
or_result = boolean_retrieval_or('covid', 'vaksin')
end_time = time.time()
print(f"OR query result for covid and vaksin: {or_result} (Time taken: {end_time - start_time}s)")

start_time = time.time()
or_result = boolean_retrieval_or('vaksin', 'indonesia')
end_time = time.time()
print(f"OR query result for vaksin and indonesia: {or_result} (Time taken: {end_time - start_time}s)")

# NOT Queries
start_time = time.time()
not_result = boolean_retrieval_not('covid')
end_time = time.time()
print(f"NOT query result for not covid: {not_result} (Time taken: {end_time - start_time}s)")

start_time = time.time()
not_result = boolean_retrieval_not('indonesia')
end_time = time.time()
print(f"NOT query result for not indonesia: {not_result} (Time taken: {end_time - start_time}s)")

# Record time for creating inverted index
start_time = time.time()
inverted_index = defaultdict(list)
for doc_id, content in enumerate(contents):
    terms = preprocess_text(content)
    for term in set(terms):
        inverted_index[term].append(doc_id)
end_time = time.time()
index_creation_time = end_time - start_time
print(f"Time taken to create inverted index: {index_creation_time}s")
