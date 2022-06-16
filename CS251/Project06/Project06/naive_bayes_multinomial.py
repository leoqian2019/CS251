'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Leo Qian
CS 251 Data Analysis Visualization, Fall 2021
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        self.num_classes = num_classes
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.log_class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.log_class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        the log of the class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        log of the class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.log_class_priors and self.log_class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape

        counts = np.zeros(self.num_classes)

        for label in y:
            counts[label] += 1
        
        self.log_class_priors = np.log(counts/num_samps)

        count_for_samples = np.zeros((self.num_classes,num_features))

        for i in range(num_samps):
            for j in range(num_features):
                count_for_samples[y[i]][j] += data[i][j]
        
        self.log_class_likelihoods = np.log((count_for_samples+1)/(count_for_samples.sum(axis = 1,keepdims=True)+num_features))



    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops using matrix multiplication or with a loop and
        a series of dot products.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: can be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        num_test_samps, num_features = data.shape

        prediction = np.zeros(num_test_samps)
        for i in range(num_test_samps):
            log_posterior = np.zeros(self.num_classes)
            for j in range(self.num_classes):
                log_posterior[j] = self.log_class_priors[j] + (data[i,:]*self.log_class_likelihoods[j,:]).sum()
            
            max_index = np.argmax(log_posterior)
            prediction[i] = max_index
        
        return prediction.astype(int)

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''

        return (y==y_pred).sum()/y.shape[0]

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.
        num_classes = len(np.unique(y))

        matrix = np.zeros((num_classes,num_classes))

        for i in range(y.shape[0]):
            matrix[y[i].astype(int)][y_pred[i].astype(int)] += 1
        
        return matrix