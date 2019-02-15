import numpy as numpy
import math

learnSet = numpy.genfromtxt("dataset1.csv", delimiter=";", usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                            converters={5: lambda s: 0 if s == b"-1" else float(s),
                                        7: lambda s: 0 if s == b"-1" else float(s)})
validationSet = numpy.genfromtxt("validation1.csv", delimiter=";", usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                                 converters={5: lambda s: 0 if s == b"-1" else float(s),
                                             7: lambda s: 0 if s == b"-1" else float(s)})


def GetDistance(learnSet, testSet):
    distance = []  # Creates an empty list(2d array)
    for rowId in range(len(learnSet)):  # For all elements in learnSet
        sum = 0  # Sets the sum of distances to 0
        for i in range(1, learnSet.shape[1]):  # For every element(form 1) in the seconde row of the 2d array
            sum += numpy.abs(testSet[i] - learnSet[rowId][i]) ** 2
            # Calculates the difference between the givven data point and existing data
        distance.append([learnSet[rowId][0], numpy.sqrt(sum)])
        # Adds the id and Pythagorean theorem result to the 2d array
    return distance


def GetLabels(nearest):
    labels = []
    for index in range(len(nearest)):  # For every element in the list that was givven
        label = nearest[index][
                    0] % 10000  # The modulo is used to be able to work, on data different than data form 2000
        if label < 301:  # If the label is before first of March
            labels.append('winter')  # The label will be marked a 'winter'
        elif 301 <= label < 601:  # If the label is between first of March and first of June
            labels.append('lente')  # The label will be marked a 'lente'
        elif 601 <= label < 901:  # If the label is between first of June and first of September
            labels.append('zomer')  # The label will be marked a 'zomer'
        elif 901 <= label < 1201:  # If the label is between first of September and first of December
            labels.append('herfst')  # The label will be marked a 'herfst'
        else:  # If the label is after first of December
            labels.append('winter')  # The label will be marked a 'winter'
    return labels


def MostCommonInList(list):
    return max(set(list), key=list.count,
               default=0)  # Gets the most common element(season) in the list and returns this element.


def GetMaxMin(learnSet):
    maxmin = [[1, 0]] + [[-math.inf, math.inf] for i in range(1, learnSet.shape[1])]
    # List of elements with max and min values
    for rowId in range(len(learnSet)):  # For all elements in learnSet
        for i in range(1, learnSet.shape[1]):  # For every element(form 1) in the seconde row of the 2d array
            if maxmin[i][0] < learnSet[rowId][i]:  # Check max
                maxmin[i][0] = learnSet[rowId][i]  # Set new max
            if maxmin[i][1] > learnSet[rowId][i]:  # Check min
                maxmin[i][1] = learnSet[rowId][i]  # Set new min
    return maxmin


def GenerateCentroids(learnSet, k):
    centroids = []  # List of data for cluster points
    maxmin = GetMaxMin(learnSet)  # Get limiters for data board
    for clusterIndex in range(k):  # Loop k times clusters
        centroids.append(
            [numpy.random.randint(maxmin[i][1], maxmin[i][0]) for i in range(len(maxmin))])  # Randomize cluster pint
        centroids[-1][0] = clusterIndex
    return numpy.array(centroids)  # Convert to numpy array


def FindNearestCentroid(centroids, dataSet):
    dataWithClusters = []  # List of data points with the distance
    for data in dataSet:
        dataWithClusters.append([
            data,
            GetDistance(centroids, data)
        ])
    return dataWithClusters


def MakeClusters(dataWithClusters, k):
    clusters = {i: [] for i in range(k)}
    for data in dataWithClusters:
        clusterArray = data[1]
        clusterArray.sort(key=lambda x: x[1])
        clusters[clusterArray[0][0]].append([clusterArray[0], data[0]])
    return list(clusters.values())


def RecalculateCentroidPosition(clusters, learnSet):
    centroids = []
    for cluster in clusters:
        if len(cluster) is 0:
            continue
        centroid = [cluster[0][0][0]] + ([0] * (learnSet.shape[1] - 1))  # Is 7 + 1
        for data in cluster:
            for elIndex in range(1, learnSet.shape[1]):  # Every element in original point data
                centroid[elIndex] += data[1][elIndex]
        for elIndex in range(1, learnSet.shape[1]):
            centroid[elIndex] = centroid[elIndex] / len(cluster)
        centroids.append(centroid)
    return numpy.array(centroids)


def CheckDifferencial(distentces):
    if len(distentces) < 3:  # Minimum of 3 entries because of double differential
        return 0
    return numpy.diff(distentces, n=2)[-1]


def CalculateBestK(learnSet, nRecalc):
    distentces = []
    for k in range(2, len(learnSet) - 1):
        centroids = CalculateBestCentroids(GenerateCentroids(learnSet, k), learnSet, k, nRecalc)
        clusters = MakeClusters(FindNearestCentroid(centroids, learnSet), k)
        dist = 0
        for cluster in clusters:
            if len(cluster) > 0:
                for data in cluster[0]:
                    dist += data[1]
        distentces.append(dist)
        if CheckDifferencial(distentces) < 0:
            return k - 1


def CalculateBestCentroids(centroids, learnSet, k, nRecalc):
    for i in range(nRecalc):
        centroids = RecalculateCentroidPosition(MakeClusters(FindNearestCentroid(centroids, learnSet), k), learnSet)
    return centroids


for i in range(50):
    print(CalculateBestK(learnSet, 1))
