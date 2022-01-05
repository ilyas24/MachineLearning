import numpy as np

class KMeans:
    def __init__(self, init_centers):
        """ This class represents the K-means model.

        For the following:
        - N: Number of samples.
        - D: Dimension of input features.
        - K: Number of centers.
             NOTE: K > 1

        Args:
        - init_centers (ndarray (shape: (K, D))): A KxD matrix consisting K D-dimensional centers.
        """

        assert len(init_centers.shape) == 2, f"init_centers should be a KxD matrix. Got: {init_centers.shape}"
        (self.K, self.D) = init_centers.shape
        assert self.K > 1, f"There must be at least 2 clusters. Got: {self.K}"

        # Shape: K x D
        self.centers = np.copy(init_centers)

    def train(self, train_X, max_iterations=1000):
        """ This method trains the K-means model.

        NOTE: This method updates self.centers

        The algorithm is the following:
        - Assigns data points to the closest cluster center.
        - Re-computes cluster centers based on the data points assigned to them.
        - Update the labels array to contain the index of the cluster center each point is assigned to.
        - Loop ends when the labels do not change from one iteration to the next. 

        Args:
        - train_X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional input data.
        - max_iterations (int): Maximum number of iterations.

        Output:
        - labels (ndarray (shape: (N, 1))): A N-column vector consisting N labels of input data.
        """
        assert len(train_X.shape) == 2 and train_X.shape[1] == self.D, f"train_X should be a NxD matrix. Got: {train_X.shape}"
        assert max_iterations > 0, f"max_iterations must be positive. Got: {max_iterations}"
        N = train_X.shape[0]

        labels = np.empty(shape=(N, 1), dtype=np.long)
        distances = np.empty(shape=(N, self.K))
        
        for _ in range(max_iterations):
            old_labels = labels

            # to store cluster
            clusters = {}
            for index in range(self.K):
                clusters[index] = []
        
            i = 0
            # loop until all rows have been checked
            while (i < N):
                # start off min with first value in row
                closest = np.sqrt(np.sum(np.square(train_X[i] - self.centers[0])))
                closest_index = 0
                for j in range(self.K):
                    # euclidean distance
                    distances[i][j] = np.sqrt(np.sum(np.square(train_X[i] - self.centers[j])))
                    # if there's a closer distance, change current closest distance
                    if distances[i][j] < closest:
                        closest = distances[i][j]
                        closest_index = j
                        
                clusters[closest_index].append(train_X[i])
                i = i + 1
            
            # re-compute clusters
            for k in range(self.K):
                # set center with mean of cluster
                self.centers[k] = np.mean(clusters[k], axis=0)
               
            # update label array
            labels = np.argmin(distances, axis=1).reshape((N, 1))
            count = count + 1
            # Check convergence
            if np.allclose(old_labels, labels):
                break
        
        return labels

'''
centres = np.matrix([[1,0,5],[2,3,9],[6,6,8],[1,1,1]])
train_x = np.matrix([[1,0,5],[2,3,9],[6,6,8],[1,1,5], [1,1,1], [2,3,9]])

print(centres)
print(centres.shape)
print(abs(train_X[i] - self.centers[j])**2)

#print(train_x[0] - centres[0])
#print(np.sqrt(np.sum(np.square(train_x[0] - centres[1]))))
K_m = KMeans(centres)
print(K_m.centers)
l = K_m.train(train_x)
K_m.train(train_x)
print(K_m.centers)
print("Labels", l)
print(l.shape)
'''