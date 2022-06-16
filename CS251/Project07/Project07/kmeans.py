'''kmeans.py
Performs K-Means clustering
Leo Qian
CS 251 Data Analysis Visualization, Fall 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data


    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance = np.sqrt(((pt_1-pt_2)**2).sum())

        return distance

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.
        
        

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        distance = np.sqrt(((centroids-pt)**2).sum(axis=1))

        return distance

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        indices = np.arange(self.num_samps)
        choice = np.random.choice(indices,k,False)
        self.centroids = self.data[choice,:]
        self.k = k

        return self.centroids

    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all 
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        old_centroid = self.initialize(k)
        old_labels = self.assign_labels(old_centroid)

        i = 0
        diff = 9223372036854775807
        while i <= max_iter and diff > tol:
            new_centroid,centroid_diff = self.update_centroids(k, old_labels, old_centroid)
            diff = (centroid_diff.sum(axis = 1)).max()
            old_centroid = new_centroid
            old_labels = self.assign_labels(old_centroid)
            i += 1
            if verbose == True:
                print(new_centroid)
                print(diff)
        self.data_centroid_labels = old_labels
        self.centroids = old_centroid

        self.inertia = self.compute_inertia()


        return self.inertia,i


    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        best_centroid = np.zeros((k,self.num_features))
        best_label = np.zeros(self.num_samps)
        best_inertia = 9223372036854775807

        for i in range(n_iter):
            inertia,_ = self.cluster(k,verbose=verbose)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroid = self.centroids
                best_label = self.data_centroid_labels
        
        self.centroids = best_centroid
        self.data_centroid_labels = best_label
        self.inertia = best_inertia

    def assign_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = []

        for data in self.data:
            distances = self.dist_pt_to_centroids(data, centroids)
            index = 0
            for i in range(distances.shape[0]):
                if i > index and distances[i] < distances[index]:
                    index = i
            
            labels.append(index)
        
        self.data_centroid_labels = np.asarray(labels)
        
        return np.asarray(labels)


    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        sum = np.zeros((k,self.num_features))
        count = np.zeros((k,1))

        for i in range(self.num_samps):
            sum[data_centroid_labels[i],:] += self.data[i]
            count[data_centroid_labels[i]] += 1
        
        for i in range(k):
            if count[i][0] == 0:
                count[i][0] = 1
                sum[i] = self.data[np.random.randint(0,self.num_samps)]

        new_centroid = sum/count
        # print(new_centroid)

        centroid_diff = new_centroid-prev_centroids

        return new_centroid,centroid_diff
        


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        sum = 0
        for i in range(self.num_samps):
            sum += ((self.data[i]-self.centroids[self.data_centroid_labels[i]])**2).sum()
        
        sum /= self.num_samps

        return sum
        

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        plt.scatter(self.data[:,0],self.data[:,1],c=self.data_centroid_labels,cmap=cartocolors.sequential.DarkMint_7.mpl_colormap)

        plt.scatter(self.centroids[:,0], self.centroids[:,1],marker="<",s=50,c="Black")

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: number of iteration to run clustering with different initial condition

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertia_list = []
        for i in range(max_k):
            # inertia,i = self.cluster(i+1)
            self.cluster_batch(i+1,n_iter=n_iter)
            inertia_list.append(self.inertia)

        k_range = np.arange(max_k)+1
        plt.plot(k_range,inertia_list)
        plt.xlabel("cluster number(k)")
        plt.ylabel("inertia")
        plt.xticks(k_range)
        



    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        pass
