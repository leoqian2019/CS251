'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
YOUR NAME HERE
CS 251 Data Analysis Visualization, Fall 2021
'''
import re
import os
import numpy as np


def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def count_words(email_path='data/enron/'):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    frequency = {}

    all_ham_files = os.listdir(email_path+"ham")
    all_spam_files = os.listdir(email_path+"spam") 

    for textFile in all_ham_files:

        f = open(email_path+"ham/"+textFile,"r")

        for line in f:
            words = tokenize_words(line)
            for word in words:
                if frequency.get(word) == None:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
        f.close()

    for textFile in all_spam_files:

        f = open(email_path+"spam/"+textFile,"r")

        for line in f:
            words = tokenize_words(line)
            for word in words:
                if frequency.get(word) == None:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
        
        f.close()
    
    return frequency,len(all_ham_files)+len(all_spam_files)


def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''
    
    map_result = np.asarray(list(word_freq.items()))
    newList = map_result[:,1].astype(np.int64)


    indices = np.argsort(newList)[::-1]

    if map_result.shape[0] < 200:
        num_features = map_result.shape[0]

    top_words = []
    count = []
    for index in indices[:num_features]:
        top_words.append(map_result[index][0])
        count.append(newList[index])
    
    return top_words,count




def make_feature_vectors(top_words, num_emails, email_path='data/enron/'):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    feats = np.zeros((num_emails,len(top_words)))
    y = np.empty(num_emails)
    

    all_ham_files = os.listdir(email_path+"ham")
    all_spam_files = os.listdir(email_path+"spam") 

    for i in range(len(all_ham_files)):

        f = open(email_path+"ham/"+all_ham_files[i],"r")

        for line in f:
            words = tokenize_words(line)
            for word in words:
                if word not in top_words:
                    continue
                feats[i][top_words.index(word)] += 1
        f.close()
        y[i] = 1

    for i in range(len(all_spam_files)):

        f = open(email_path+"spam/"+all_spam_files[i],"r")

        for line in f:
            words = tokenize_words(line)
            for word in words:
                if word not in top_words:
                    continue
                feats[i+len(all_ham_files)][top_words.index(word)] += 1
        
        f.close()

        y[i+len(all_ham_files)] = 0
    
    return feats,y




def make_train_test_sets(features, y, test_prop=0.2, shuffle=True):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]
    
    x_train = features[:int(features.shape[0]*(1-test_prop)),:]
    y_train = y[:int(features.shape[0]*(1-test_prop))]
    inds_train = inds[:int(features.shape[0]*(1-test_prop))]

    x_test = features[int(features.shape[0]*(1-test_prop)):,:]
    y_test = y[int(features.shape[0]*(1-test_prop)):]
    inds_test = inds[int(features.shape[0]*(1-test_prop)):]

 

    return x_train,y_train,inds_train,x_test,y_test,inds_test


    # Your code here:

