import numpy as numpy

learnSet = numpy.genfromtxt("dataset1.csv", delimiter=";", usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
validationSet = numpy.genfromtxt("validation1.csv", delimiter=";", usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
days = numpy.genfromtxt("days.csv", delimiter=";", usecols=[0,1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

def GetDistance(learnSet, testSet):
        distance = [] #Creates an empty list(2d array)
        for rowId in range(len(learnSet)): #For all elements in learnSet
                sum = 0 #Sets the sum of distances to 0
                for i in range(1,learnSet.shape[1]): #For every element(form 1) in the seconde row of the 2d array
                        sum += numpy.abs(testSet[i]-learnSet[rowId][i])**2 #Calculates the difference between the givven data point and existing data
                distance.append([learnSet[rowId][0],numpy.sqrt(sum)]) #Adds the id and Pythagorean theorem result to the 2d array
        return distance

def GetNearest(distances, k):
        nearest = [] #Creates an empty list(2d array)
        distances.sort(key=lambda x: x[1]) #Sorts distances
        if k < len(distances): #Checks if K is smaller then length of distances
                for i in range(k): #For i in the range of 0 to K
                        nearest.append([distances[i][0],distances[i][1]]) #Adds the id and distance to the list
        return nearest

def GetLabels(nearest):
        labels = []	
        for index in range(len(nearest)): #For every element in the list that was givven
                label = nearest[index][0]%10000 #The modulo is used to be able to work, on data different than data form 2000
                if label < 301:  #If the label is before first of March
                        labels.append('winter') #The label will be marked a 'winter'
                elif label >= 301 and label < 601: #If the label is between first of March and first of June
                        labels.append('lente') #The label will be marked a 'lente'
                elif label >= 601 and label < 901: #If the label is between first of June and first of September
                        labels.append('zomer') #The label will be marked a 'zomer'
                elif label >= 901 and label < 1201: #If the label is between first of September and first of December
                        labels.append('herfst') #The label will be marked a 'herfst'
                else:   #If the label is after first of December
                        labels.append('winter') #The label will be marked a 'winter'
        return labels

	
def MostCommonInList(list):
    return max(set(list), key=list.count, default=0) #Gets the most common element(season) in the list and returns this element. 

def Validater(learnSet, validationSet, k):
       numberOfCorrectGuesses = 0  #Make var and set it to 0
       listOfCorrectlabels = GetLabels(validationSet) #Gets the correct labels for the givven days
       for i in range(len(validationSet)): #For every element in the validation set
               if MostCommonInList(GetLabels(GetNearest(GetDistance(learnSet, validationSet[i]), k))) == listOfCorrectlabels[i]: #If the guessed label is equal to the correct label
                        numberOfCorrectGuesses += 1 #Then Up the count of correct geusses
       return  (100/len(validationSet)*numberOfCorrectGuesses) #Get procentage of correctness

def ValidateRange(learnSet, validationSet, beginK, stopK):
        if beginK <= 0: #If the givven K is smaller than 01
                beginK = 0 #the K will be 0
        if stopK >= len(learnSet): #If the givven K is bigger than the length of the learn set
                stopK = len(learnSet) #The K will become the length of the learn set
        for i in range(beginK,stopK): #For every K inbetween beginK and stopK
                print(f"{i}: {Validater(learnSet, validationSet, i)}") #Print procentage of correct guesses

def GuessDays(learnSet, days, k):
        for i in range(len(days)):
                print(f"Day {i} is a {MostCommonInList(GetLabels(GetNearest(GetDistance(learnSet, days[i]), k)))} day.") #Best guess of a givven day will be calculated and printed


print("==============================")
GuessDays(learnSet, days, 59)                           #Geusses the givven days with a K of 59
print("==============================")
print(f"58: {Validater(learnSet, validationSet, 58)}")  #Caluclates the correctness with a K of 58 
print("==============================")
ValidateRange(learnSet, validationSet, 54, 64)          #Caluclates the correctness with a K of 54 to 64
print("==============================")