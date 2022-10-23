from numpy import *
import operator
import loadMINST
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key =operator.itemgetter(1), reverse=True)
    #print sortedClassCount
    return sortedClassCount[0][0]
 
def handwritingClassTest():
    trainingMat, hwLabels, size = loadMINST.load('/Users/phyllislee/VScode/AI/KNN/pymnist/train-images-idx3-ubyte','/Users/phyllislee/VScode/AI/KNN/pymnist/train-labels-idx1-ubyte')
    dataUnderTest, classNumStr, size = loadMINST.load('/Users/phyllislee/VScode/AI/KNN/pymnist/t10k-images-idx3-ubyte','/Users/phyllislee/VScode/AI/KNN/pymnist/t10k-labels-idx1-ubyte')
    errorCount = 0.0
    for i in range(size):
        classifierResult = classify0(dataUnderTest[i,:], trainingMat, hwLabels, 3)
        print("the NO.%d classifier came back with: %d, the real answer is: %d, error count is: %d" % (i, classifierResult, classNumStr[i], errorCount))
        if (classifierResult != classNumStr[i]): errorCount +=+ 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(size)))

