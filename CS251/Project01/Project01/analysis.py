'''analysis.py
Run statistical analyses and plot Numpy ndarray data
Leo Qian
CS 251 Data Analysis Visualization, Fall 2021
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        dataset = self.data.select_data(headers,rows)

        return dataset.min(axis=0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        dataset = self.data.select_data(headers,rows)

        return dataset.max(axis=0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: Loops are forbidden!
        '''
        minimum = self.min(headers,rows)

        maximum = self.max(headers,rows)

        return [minimum,maximum]

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: Loops are forbidden!
        '''
        dataset = self.data.select_data(headers,rows)

        # if the row list isn't empty, then calculate the mean base on how many rows there are
        # else, calculate the mean base on how many samples we have
        if len(rows) > 0:
            return dataset.sum(axis = 0)/len(rows)
        else:

            return dataset.sum(axis = 0)/self.data.get_num_samples()
        # return dataset.mean(axis = 0)

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: Loops are forbidden!
        '''
        dataset = self.data.select_data(headers,rows)

        # if the row list is empty, can calculate the variance using the number of data samples we have
        # the function I used for variance is first canculate the square of the difference between each value and the mean, 
        # then canculate the sum of each column; in the end, divide it by the total number of samples we have
        if len(rows) == 0:
            return ((dataset-self.mean(headers,rows))**2).sum(axis=0)/self.data.get_num_samples()
        else:
            return ((dataset-self.mean(headers,rows))**2).sum(axis=0)/len(rows)
        
        return var

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: Loops are forbidden!
        '''
        # take the squre root of the variance
        return np.sqrt(self.var(headers,rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        
        # first find the np arrays for both the x and y data
        x_array = self.data.select_data([ind_var])
        
        y_array = self.data.select_data([dep_var])
        
        
        # then, plot it with a scatter plot and indicate the correct title
        plt.scatter(x_array,y_array)
        plt.title(title)

        # return the x and y np array
        return x_array,y_array
    
    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        # first construct subplots of the correct amount and size, and indicate the sharex and sharey parameter
        fig, axes = plt.subplots(len(data_vars), len(data_vars),sharex='col',sharey="row",figsize=fig_sz)
        # indicate the general title for the whole subplot
        plt.suptitle(title)

        plt.rcParams["figure.autolayout"] = True

        # initialize the x y indices for the subplots
        x = 0
        y = 0
        for data1 in data_vars:
            # start to change the x index 
            x += 1
            for data2 in data_vars:
                # find the correct arrays of data
                x_array = self.data.select_data([data1])
                y_array = self.data.select_data([data2])

                # plot the array in the correct subplot
                plt.subplot(len(data_vars),len(data_vars),x+y)
                plt.plot(x_array,y_array)

                # label the column on the left of the subplots
                if (x+y-1)%len(data_vars) == 0:
                    plt.ylabel(data2)
                
                # label all the x axis of the rows on the bottom of the subplot
                if x == len(data_vars):
                    plt.xlabel(data1)

                y += 1
                
            
            # change the y index to make sure the subplot index is correct
            y -= 1
            
        
        return fig,axes
        

    
