# A simple Naive Bayes spam classifier created on the basis of a Deep Learning
# in Python class (Machine Learning, Data Science and Deep Learning with
# Python) on Udemy -- https://www.udemy.com/course/data-science-and-machine-
# learning-with-python-hands-on/

# Import the required Python libraries

import os
import io
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create a function to read in the training data

def readFiles(path):
    # repeat these steps for all the files in the filenames list
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            # Ignore header data in the emails
            inBody = False
            # Create an empty list to store the text data by line
            lines = []
            # Read in the file with single byte ASCII character encoding
            f = io.open(path, 'r', encoding='latin1')
            # Go through line by line and append each line to the created
            # list if the line is in the body of the email.
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            # Close the file
            f.close()
            # Create a string called message and concatenate in each line with
            # a line return separating it from the next
            message = '\n'.join(lines)
            # Return the filepath and the message.
            yield path, message

# Create a DataFrame containing one email per row and using the
# filename as an index
def dataFrameFromDirectory(path, classification):
    # First, create two empty parallel arrays
    rows = []
    index = []
    # repeat these actions for each message
    for filename, message in readFiles(path):
        # Add each message to a dictionary to store classification (spam or
        # ham) for each message (row in DataFrame)
        rows.append({'message': message, 'class': classification})
        # Use the filename as an index
        index.append(filename)
        
    # Returns a DataFrame with all the read emails and their classification
    return DataFrame(rows, index=index)

# Create an empty data frame to populate with the training data
data = DataFrame({'message': [], 'class': []})

# Add the training data to the empty data frame with the correct
# classification using the above function
data = data.append(dataFrameFromDirectory('emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('emails/ham', 'ham'))

# Now let's take a peek at the training data frame to check its format and result
print(data.head())

# Create a count vectozier to tokenize the words in the messages
vectorizer = CountVectorizer()
# Extract tokens to a matrix storing their frequency
counts = vectorizer.fit_transform(data['message'].values)

# Instantiate a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
# Add all the supervised classifications from the dataframe to a document object
targets = data['class'].values
# Create a model based on the training data, mapping messages to classifications
classifier.fit(counts, targets)

# Try some input data!
email_in = ''
while email_in != "quit":
    # Creates an empty list of text documents to classify. Note that at this
    # time, in order to accept user input, I have opted to only receive one
    # e-mail to verify at a time.
    text_list = []
    # Input the email body
    email_in = input('Paste an e-mail body here\nto see if it is spam or ham.\n(To quit at any time, enter \'quit\'): ')
    # Check to see if the user wants to exit
    if email_in == 'quit':
        break
    else:
        # Add the text as a document to the text_list array object, then
        # tokenize the words and obtain frequency using transform. Print out
        # the predictions obtained from the model the text was fit to given
        # the training data
        text_list.append(email_in)
        text_counts = vectorizer.transform(text_list)
        predictions = classifier.predict(text_counts)
        print(predictions)
