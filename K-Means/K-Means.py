import numpy
import math

#Get data in var
dataSet = numpy.genfromtxt("dataset1.csv", delimiter=";", usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                            converters={5: lambda s: 0 if s == b"-1" else float(s),
                                        7: lambda s: 0 if s == b"-1" else float(s)})


def GetDistance(dataSet, testSet):                                # Gets dataSet and testSet, and compairs thier distances
    distance = []                                                 # Creates an empty list(2d array)
    for rowId in range(len(dataSet)):                             # For all elements in dataSet
        sum = 0                                                   # Sets the sum of distances to 0
        for i in range(1, dataSet.shape[1]):                      # For every element(form 1) in the seconde row of the 2d array
            sum += numpy.abs(testSet[i] - dataSet[rowId][i]) ** 2 # Calculates the difference between the givven data point and existing data
        distance.append([dataSet[rowId][0], numpy.sqrt(sum)])     # Adds the id and Pythagorean theorem result to the 2d array
    return distance                                               # Returns list of id's and distances


def GetDataRange(dataSet):
    dataRange = [[1, 0]] + [[-math.inf, math.inf] for i in range(1, dataSet.shape[1])] # List of elements with max and min values
    for rowId in range(len(dataSet)):                # For all elements in dataSet
        for i in range(1, dataSet.shape[1]):         # For every element(form 1) in the seconde row of the 2d array
            if dataRange[i][0] < dataSet[rowId][i]:  # Check if the element is above the current maximum
                dataRange[i][0] = dataSet[rowId][i]  # Then sets the current maximum to the elements data
            if dataRange[i][1] > dataSet[rowId][i]:  # Check if the element is below the current minimum
                dataRange[i][1] = dataSet[rowId][i]  # Then sets the current minimum to the element data
    return dataRange


def GenerateCentroids(dataSet, k):                          # Generates centroids randomly(inside of the dataSet's boundaries)
    centroids = []                                          # List of centroids.
    dataRange = GetDataRange(dataSet)                       # Get the boundaries to pick between
    for clusterIndex in range(k):                           # Loops K times, so there will be K centroids
        centroids.append([numpy.random.randint(dataRange[i][1], dataRange[i][0]) for i in range(len(dataRange))])                                                  # Generates the centroids randomly(inside of the dataSet's boundaries)
        centroids[-1][0] = clusterIndex                     # Gives the (last)centroid an id
    return numpy.array(centroids)                           # Returns a numpy array of centroids(with id's) 


def indexDistanceToCentroids(centroids, dataSet):           # Determines for each data point what it's distance is to each centroid
    dataWithClusters = []                                   # List of data points with the distance
    for data in dataSet:                                    # For each element in dataSet
        dataWithClusters.append([               
            data,                                           # Add the data
            GetDistance(centroids, data)                    # And the distance to the centroids
        ])
    return dataWithClusters                                 # Returns a list filled with data and it's distance to every centroid


def MakeClusters(dataWithClusters, k):                      # Makes clusters
    clusters = {i: [] for i in range(k)}                    # Create a dictonary the size of the number of centroids
    for data in dataWithClusters:                           # For every element in dataWithClusters
        clusterArray = data[1]                              # Get distances to every centroid of this data point
        clusterArray.sort(key=lambda x: x[1])               # Sort on distances
        clusters[clusterArray[0][0]].append([clusterArray[0], data[0]]) # Add data point and centroid id to clustter 
    return list(clusters.values())                          # Returna list of clusters with data points


def RecalculateCentroidPosition(clusters, dataSet):                     # Recalculates the centroids positions based on the clusters
    centroids = []                                                      # Make centroid list
    for cluster in clusters:                                            # For every element in clusters
        if len(cluster) is 0:                                           # If the element is empty
            continue                                                    # Then skip this element
        centroid = [cluster[0][0][0]] + ([0] * (dataSet.shape[1] - 1))  # Centroid is a empty cluster plus (The number of sets in the dataset) empty elements
        for data in cluster:                                            # For every element in cluster
            for elIndex in range(1, dataSet.shape[1]):                  # Every element in range of the number of sets in the dataset
                centroid[elIndex] += data[1][elIndex]                   # Add data[1] to centroid(datapoint data)
        for elIndex in range(1, dataSet.shape[1]):                      # Every element in range of the number of sets in the dataset
            centroid[elIndex] = centroid[elIndex] / len(cluster)        # The everage point is all the data combined defided by the number of entries
        centroids.append(centroid)                                      # Add calculated centroid to centroid list
    return numpy.array(centroids)                                       # Returns numoy array of recaculated centroids



def CalculateBestCentroids(centroids, dataSet, k, numberOfRecalculations): # This function recalculates centoriods numberOfRecalculations times. (For a given K)
    for _i in range(numberOfRecalculations):                                # For numberOfRecalculations times
        centroids = RecalculateCentroidPosition(MakeClusters(indexDistanceToCentroids(centroids, dataSet), k), dataSet) # Recalc the centroids
    return centroids                                                       # Returns the best centroid for the given data


def CheckDifferencial(distances):             # Calcualtes the dubbel differencial for a given list
    if len(distances) < 3:                    # Minimum of 3 entries because of double differential
        return 0                               # If not then return 0
    return numpy.diff(distances, n=2)[-1]     # Returns dubbel differencial from the given list


def CalculateBestK(dataSet, numberOfRecalculations):                                            # Calculates the best K for the given dataSet
    distances = []                                                                             # Create a empty list of distances
    for k in range(2, len(dataSet) - 1):                                                        # For in range of k = 2, to k is max-1
        centroids = CalculateBestCentroids(GenerateCentroids(dataSet, k), dataSet, k, numberOfRecalculations)  # Calculate the centorids for the dataSet 
        clusters = MakeClusters(indexDistanceToCentroids(centroids, dataSet), k)                # Create Clusters
        totalDistanceOfCluster = 0                                                              # Create var that will be used to store the total distance of the clusters(Distance to centroids)
        for cluster in clusters:                                                                # For every element in clusters
            if len(cluster) > 0:                                                                # Only if the cluster isn't empty
                for data in cluster[0]:                                                         # For every emelent in cluster[0]
                    totalDistanceOfCluster += data[1]                                           # Add distance to centorid to totalDistanceOfCluster
        distances.append(totalDistanceOfCluster/k)                                               # Add total distance to distances list
        print(f"Total distance {k}: {totalDistanceOfCluster/k}")
        if CheckDifferencial(distances) < 0:                                                   # If the dubbel differencial is negative the best K is surpased 
            return k - 1                                                                    # Because of that return the previous K


def MostCommonInList(list):
    return max(set(list), key=list.count,default=0)  # Gets the most common element(season) in the list and returns this element.


def Run(dataSet, numberOfAttemps, numberOfRecalculations):               # Finds the best K for the given dataSet
    listOfKs = []                                                        # Create a list for the Ks
    for _i in range(numberOfAttemps):                                    # For the number of attemps given
        listOfKs.append(CalculateBestK(dataSet, numberOfRecalculations)) # Add the result to the list
    mostCommonK = MostCommonInList(listOfKs)                             # Get the most common in this list
    return [mostCommonK,listOfKs]                                        # Returns the most common K and the list of Ks

def Certainty(listOfKs):                                                 # Shows how Certian the AI is of it's calculations 
    uniqueKs = set(listOfKs)
    print("\nCertainty")                                                   # Gets the unique items from listOfKs
    for i in uniqueKs:                                                   # For every unique element
        print(f"{i}: {100/len(listOfKs)*listOfKs.count(i)}%")            # Print the K and the procentage of occurance


numberOfRecalculations = 30     # The bigger this var, the beter the centroids are centerd
numberOfAttemps = 10            # How many differend times random centroid will be used

resultList = Run(dataSet, numberOfAttemps, numberOfRecalculations)
print(resultList[0])

Certainty(resultList[1])
