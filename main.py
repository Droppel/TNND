from rbfnet import RBFNet
import numpy as np
from matplotlib import pyplot as plot

def main():
    outputSize = 1 # The size of the output vector
    stepSize = 1 # The amount of space between two RBF Centers
    seed = 42 # The seed
    learnRate = 0.1 # The learning rate etha
    iterations = 200 # The amount of training iterations

    np.random.seed(seed)

    trainingData = parseFile("training_data.txt")
    testData = parseFile("test_data.txt")

    # Find the smallest and biggest inputvalues. This information is used to evenly distribute the centers
    smallestInputs = np.zeros(2)
    biggestInputs = np.zeros(2)

    smallestInputs[0] = trainingData[0][0] # Initial Smallest x Value
    smallestInputs[1] = trainingData[0][1] # Initial Smallest y Value
    biggestInputs[0] = trainingData[0][0] # Initial Biggest x Value
    biggestInputs[1] = trainingData[0][1] # Initial Biggest y Value

    for patt in trainingData[1:]:
        if smallestInputs[0] > patt[0]:
            smallestInputs[0] = patt[0]
        if smallestInputs[1] > patt[1]:
            smallestInputs[1] = patt[1]

        if biggestInputs[0] < patt[0]:
            biggestInputs[0] = patt[0]
        if biggestInputs[1] < patt[1]:
            biggestInputs[1] = patt[1]

    for patt in testData:
        if smallestInputs[0] > patt[0]:
            smallestInputs[0] = patt[0]
        if smallestInputs[1] > patt[1]:
            smallestInputs[1] = patt[1]

        if biggestInputs[0] < patt[0]:
            biggestInputs[0] = patt[0]
        if biggestInputs[1] < patt[1]:
            biggestInputs[1] = patt[1]

    # Initialize the net
    rbfNet = RBFNet(stepSize, outputSize, learnRate, smallestInputs, biggestInputs)

    # Train the network, evaluate after every iteration and plot the error
    for iteration in range(iterations):
        # Evaluate
        totalError = 0
        for pattern in testData:
            totalError += rbfNet.evaluate(pattern)
        totalError /= len(testData)
        plot.scatter(iteration, totalError)
        
        #Train
        for pattern in trainingData:
            rbfNet.train(pattern)

    plot.show()
        



def parseFile(filename):
    patterns = []
    
    with open(filename) as file:
        commentOffset = 0 # We need to offset the commentlines when adding to the array
        for patt, line in enumerate(file):
            pattern = []
            if line[0] == "#":
                commentOffset += 1
                continue
            for indx, value in enumerate(line.split()):
                pattern.append(value)    
            patterns.append(pattern)
    return np.array(patterns, dtype=int)

if __name__ == '__main__':
    main()