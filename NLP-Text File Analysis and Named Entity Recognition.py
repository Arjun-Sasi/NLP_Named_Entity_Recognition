

# Below we import  necessary libraries
import os
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt

# Setting the directory path
dir_path = os.path.join(os.getcwd())

# Initializing lists to hold the data and labels
data = []
labels = []

# Looping through the files in the directory
for file in os.listdir(dir_path):
    # Checking if the file is a text file
    if file.endswith(".txt"):
        # Open the file and read the contents
        with open(os.path.join(dir_path, file), "r") as f:
            contents = f.read()
            # Cleaning and preparing the data
            # Replacing newline characters with spaces
            contents = contents.replace("\n", " ")
            #  Remove punctuation
            contents = re.sub(r'[^\w\s]', '', contents)
            #  Convert to lowercase
            contents = contents.lower()
            # Tokenize the contents
            tokens = nltk.word_tokenize(contents)
            #  Remove stop words
            stop_words = nltk.corpus.stopwords.words("english")
            tokens = [x for x in tokens if x not in stop_words]
            # Stem the tokens
            stemmer = nltk.stem.PorterStemmer()
            tokens = [stemmer.stem(x) for x in tokens]
            # Adding the data and label to the lists
            data.append(tokens)
            labels.append(file.split(".")[0])
# Printing some basic info about the dataset
print("Number of files:", len(data))
print("Number of tokens:", sum([len(x) for x in data]))
print("Number of labels:", len(labels))
print("Unique labels:", set(labels))


# ### Description 
# 
# The above code reads in a collection of text files from a specified directory and processes the contents of each file to create a dataset for text classification.The code first imports several libraries that are used throughout the script, including os, re, nltk, numpy, and matplotlib.Next, the code sets the directory path to the current working directory and initializes two empty lists, data and labels, to hold the processed data and labels respectively.The code then enters a loop that iterates over the files in the specified directory. For each file, the code checks if the file is a text file (i.e., if it ends with the .txt extension) and, if so, it reads in the contents of the file and performs some cleaning and preparation on the data.First, the code replaces newline characters with spaces and removes punctuation from the text. It then converts the text to lowercase and tokenizes it into a list of individual words (tokens).Next, the code removes any stop words (common words that do not provide much information, such as "the" and "a") from the list of tokens using the stopwords corpus from nltk.
# The code then applies stemming to the list of tokens, which means reducing each word to its base form by removing suffixes. This is done using the PorterStemmer from nltk.Finally, the code adds the processed data (i.e., the list of stemmed tokens) and the label (i.e., the name of the file, minus the .txt extension) to the data and labels lists, respectively.After the loop has completed, the code prints some basic information about the dataset, including the number of files, the total number of tokens, the number of labels, and the unique labels in the dataset

# ## Exploratory Data Analysis
# 

# In[2]:


# Count the frequency of each token in the dataset
token_counts = {}
for tokens in data:
    for token in tokens:
        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1
# Print the top 10 most common tokens
print("Top 10 most common tokens:")
for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(token, count)


# ### Description 
# 
# The above code is performing some exploratory data analysis on the dataset by counting the frequency of each token and printing the top 10 most common tokens.To do this, the code first initializes an empty dictionary called token_counts which will be used to store the frequency of each token. It then loops through the list of data (which is a list of lists of tokens) and for each token, it checks if it is already in the token_counts dictionary. If it is not, it adds the token to the dictionary with a count of 1. If it is already in the dictionary, it increments the count by 1.After the loop has finished, the code sorts the token_counts dictionary by the count of each token in descending order and prints the top 10 most common tokens. It does this by looping through the sorted dictionary and printing the token and its count.Overall, this code helps us understand the most common tokens used in the dataset and can give us an idea of the types of words that are most important or relevant in the text data.

# ## Descriptive Analytics
# 

# In[3]:


# Question 1: What is the distribution of labels in the dataset?
# Here we check the balance of classes in the dataset, which can impact the performance of machine learning models.
print("\nLabel distribution:")
label_counts = {}
for label in labels:
    if label not in label_counts:
        label_counts[label] = 1
    else:
        label_counts[label] += 1
for label, count in label_counts.items():
    print(label, count)


# Getting the labels and their counts
labels, counts = zip(*label_counts.items())

# Creating the bar chart
plt.bar(labels, counts)

# Title and axis labels
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")

#  show the plot
plt.show()

# Question 2: What are the most common tokens used in each label?
# Here we gain an interesting insight to the data because it will give us an idea of the types of tokens that are most indicative of each label.
print("\nMost common tokens in each label:")
for label in set(labels):
    # Get the data for the label
    label_data = [x for i, x in enumerate(data) if labels[i] == label]
    # Count the frequency of each token in the label data
    token_counts = {}
    for tokens in label_data:
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 1
            else:
                token_counts[token] += 1


# Set the width of the bars
bar_width = 0.35

# Loop through each label
for label in set(labels):
    # Get the data for the label
    label_data = [x for i, x in enumerate(data) if labels[i] == label]
    # Count the frequency of each token in the label data
    token_counts = {}
    for tokens in label_data:
        for token in tokens:
            if token not in token_counts:
                token_counts[token] = 1
            else:
                token_counts[token] += 1
    # Sort the tokens by their count
    sorted_token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    # Get the labels and counts for the tokens
    token_labels, token_counts = zip(*sorted_token_counts)
    

    token_label_dict = {}
    for pair in sorted_token_counts:
        token_label_dict[pair[0]] = pair[1]
    
    # Set the position of the bars
    bar_positions = np.arange(len(token_counts))
    # Create the bar chart
    plt.bar(bar_positions, token_counts, width=bar_width, label=label)
    # Title and axis labels
    plt.title("Most Common Tokens in Label: " + label)
    plt.xlabel("Token")
    plt.ylabel("Count")
    # Show the plot
    plt.show()

    # Print the top 10 most common tokens in the label
print("\nTop 10 most common tokens in label '{}':".format(label))

for token_final_count in sorted(token_label_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    token, count = token_final_count
    print(token,count)


# Question 3: What are the most common bigrams (two consecutive tokens) used in each label?
# This is interesting because it will give us an idea of the combinations of tokens that are most indicative of each label.
print("\nMost common bigrams in each label:")
bigram_counts = {}
for label in set(labels):
    #
    # Get the data for the label
    label_data = [x for i, x in enumerate(data) if labels[i] == label]
    # Find the bigrams in the label data
    bigrams = nltk.bigrams(label_data[0])
    # Count the frequency of each bigram in the label data

    for bigram in bigrams:
        if bigram not in bigram_counts:
            bigram_counts[bigram] = 1
        else:
            bigram_counts[bigram] += 1

        # Set the width of the bars
bar_width = 0.35

# Loop through each label
for label in set(labels):
    # Get the data for the label
    label_data = [x for i, x in enumerate(data) if labels[i] == label]
    print
# Find the bigrams in the label data
bigrams = nltk.bigrams(label_data[0])
# Count the frequency of each bigram in the label data
bigram_counts = {}
for bigram in bigrams:
    if bigram not in bigram_counts:
        bigram_counts[bigram] = 1
    else:
        bigram_counts[bigram] += 1
# Sort the bigrams by their count
sorted_bigram_counts = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)
# Get the labels and counts for the bigrams
common_bigram_dict = {}
for pair in sorted_bigram_counts:
    common_bigram_dict[pair[0]] = pair[1]

bigram_labels, bigram_counts = zip(*sorted_bigram_counts)
# Set the position of the bars
bar_positions = np.arange(len(bigram_counts))
# Create the bar chart
plt.bar(bar_positions, bigram_counts, width=bar_width, label=label)
# Title and axis labels
plt.title("Most Common Bigrams in Label: " + label)
plt.xlabel("Bigram")
plt.ylabel("Count")
# Show the plot
plt.show()

# Print the top 10 most common bigrams in the label
print("\nTop 10 most common bigrams in label '{}':".format(label))
for bigram, count in sorted(common_bigram_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(bigram, count)


# ### Description 
# 
# This code is performing some exploratory data analysis on a dataset of text files. First, it counts the frequency of each token (individual word) in the dataset and prints the top 10 most common tokens. Then, it calculates the distribution of labels (names of the text files) in the dataset and plots a bar chart to visualize the distribution. Finally, it calculates the most common tokens used in each label and plots a bar chart for each label to visualize the results. This can help us understand the characteristics of each label and the words that are most indicative of them. Additionally, the code also calculates the most common bigrams (two consecutive tokens) used in each label and plots a bar chart for each label to visualize the results. This can help us understand which pairs of words are most indicative of each label.

# ## Suggestion ( Based on the Limited Dataset )


# 1) One way to enrich the dataset is by adding additional external data. 
#For example,scraping news articles or other text data from the web and add them to the dataset. 
#This can help improve the generalizability of the model and make it more robust.

# 2) Using advanced NLP techniques: There are many advanced NLP techniques that could be applied to the dataset to extract more meaningful features. 
#For example, we could use techniques like named entity recognition (NER) to identify and extract named entities from the text.
   
  
# Here we use NER to identify and extract named entities from the text files in the directory.  
           
labels = []  # Change labels to an empty list

# Set the directory path
dirc_path = os.path.join(os.getcwd())
for file in os.listdir(dirc_path):
   # Checking if the file is a text file
   if file.endswith(".txt"):
       # Open the file and read the contents
       with open(os.path.join(dirc_path, file), "r") as f:
           contents = f.read()
           # Cleaning and preparing the data
           # Replacing the newline characters with spaces
           contents = contents.replace("\n", " ")
           # Removing punctuations
           contents = re.sub(r'[^\w\s]', '', contents)
           # Convert to lowercase
           contents = contents.lower()
           # Tokenize the contents
           tokens = nltk.word_tokenize(contents)
           # Remove stop words
           stop_words = nltk.corpus.stopwords.words("english")
           tokens = [x for x in tokens if x not in stop_words]
           # Stem the tokens
           stemmer = nltk.stem.PorterStemmer()
           tokens = [stemmer.stem(x) for x in tokens]
           
           # Use NER to identify and extract named entities from the text
           # Tagging the tokens with their part of speech
           pos_tags = nltk.pos_tag(tokens)
           # Next, we use the ne_chunk function to identify named entities
           named_entities = nltk.ne_chunk(pos_tags, binary=True)
           # Extract the named entities from the chunked tree
           named_entities = [chunk for chunk in named_entities if hasattr(chunk, "label")]
           # Flatten the named entities into a list of strings
           named_entities = [token[0] for chunk in named_entities for token in chunk]
           # Add the named entities to the data list
           data.append(named_entities)
           # Add the label to the labels list
           labels = labels + file.split((".")[0])
           


           

# To Print some basic info about the dataset
print("Number of files:", len(data))
print("Number of named entities:", sum([len(x) for x in data]))
print("Number of labels:", len(labels))
print("Unique labels:", set(labels))
       

   


# ## Description 
# 
# The above code processes a directory of text files and extracts named entities from each file using advanced NLP technique. It begins by setting up an empty list for the labels and defining the directory path.Then iterates over the files in the directory, filtering for only text files, and reads the contents of each file.The contents are cleaned and preprocessed by replacing newline characters with spaces, removing punctuations, converting to lowercase, tokenizing, and removing stop words.The tokens are then stemmed using the Porter stemmer. Next, the named entity recognition (NER) technique is applied to the tokens by first tagging them with their part of speech and then using the ne_chunk function to identify named entities.The named entities are extracted from the chunked tree,flattened into a list of strings, and added to the data list.The labels are also extracted from the file names and added to the labels list.Finally we have the code to print out some basic information about the dataset, including the number of files,the total number of named entities,the number of labels,and the unique labels.

# ## Bibliography
# 
# [1] https://www.dataknowsall.com/ner.html
# 
# 
# [2] https://uc-r.github.io/creating-text-features
# 
# [3] S. Liu, X. Wang, M. Liu, and J. Zhu, “Towards better analysis of machine learning models: A visual analytics perspective,” vol. 1, no. 1, pp. 48–56, 2017, doi: 10.1016/j.visinf.2017.01.006. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S2468502X17300086
# 
# [4] https://onlinelibrary.wiley.com/doi/full/10.1111/radm.12408
# 
# [5] https://www.kaianalytics.com/post/how-to-use-text-analysis-techniques-bring-qualitative-data-to-life
