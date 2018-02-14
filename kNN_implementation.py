import csv
import random
import math
import operator

def loadData(filename, split, trainingSet = [], testSet = []):
    with open(filename) as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in  range(len(dataset)):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

def euclideanDistance(instance1, instance2, length):
    distance_sq=0
    for i in range(length):
        distance_sq+=pow(instance1[i]-instance2[i],2)
    return math.sqrt(distance_sq)
   
def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    length=len(testInstance)-1
    for i in range(len(trainingSet)):
        dist=euclideanDistance(trainingSet[i], testInstance, length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def response(neighbors):
    classVotes={}
    for i in range(len(neighbors)):
        res=neighbors[i][-1]
        if res in classVotes:
            classVotes[res]+=1
        else:
            classVotes[res]=1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def accuracy(testSet, predictions):
    correct=0.0
    for i in range(len(testSet)):
        if testSet[i][-1]==predictions[i]:
            correct+=1.0
    return (correct/len(testSet))*100

def main():
    trainingSet=[]
    testSet=[]
    loadData('E:\\Machine_Learning\\iris_data.csv', 0.67, trainingSet, testSet)
    predictions=[]
    k=3
    for i in range(len(testSet)):
        neighbors=getNeighbors(trainingSet, testSet[i], k)
        result=response(neighbors)
        predictions.append(result)
    print(accuracy(testSet, predictions)) 

main()
