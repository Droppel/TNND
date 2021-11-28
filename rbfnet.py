from operator import le
import numpy as np
import math

# RBFNet represents an entire RBF Network
class RBFNet:

    # Initialize the network:
    # stepSize: the space between the RBF centers. (This also determines the amount of RBFNeurons)
    # outputSize: the expected outputSize
    # learnRate: the learnrate (It is the same for the entire network)
    # smallestInputs: The smallest Inputs (Used in distributing the Neurons)
    # biggestInputs: The biggest Inputs (Used in distributing the Neurons)
    def __init__(self, stepSize, outputSize, learnRate, smallestInputs, biggestInputs) -> None:
        self.outputSize = outputSize
        self.learnRate = learnRate

        halfStep = stepSize/2

        # Distribute the x and y coordinates evenly
        xCoords = np.arange(smallestInputs[0] + halfStep, biggestInputs[0] - halfStep, stepSize)
        yCoords = np.arange(smallestInputs[1] + halfStep, biggestInputs[1] - halfStep, stepSize)

        # Every x/y combination is a RBFNeurons center
        centers = []
        for x in xCoords:
            for y in yCoords:
                centers.append(np.array([x,y]))

        self.rbfSize = len(centers)

        self.rbfNeurons = []
        s_k = 0.85 * stepSize # We make s_k a bit smaller than stepsize. This causes the Neurons to cover a moderate amount
                              # of the input area
        for i in range(self.rbfSize):
            self.rbfNeurons.append(RBFNeuron(centers[i], s_k))


        self.outputNeurons = []
        for i in range(outputSize):
            self.outputNeurons.append(OutputNeuron(np.random.uniform(low=0.5, high=0.5, size=self.rbfSize)))

    # def recursiveDis(self, points, currentIndices, currentDim, resultVectors):
    #     if currentDim == 0:
    #         for i in range(len(points[currentDim])):
    #             vector = []
    #             for dim in range(len(currentIndices)):
    #                 vector.append(points[dim][currentIndices[dim]])
    #             currentIndices[currentDim] += 1
    #             resultVectors.append(np.array(vector))
    #         currentIndices[currentDim] = 0
    #     else:
    #         while True:    
    #             if currentIndices[currentDim] == len(points[currentDim]):
    #                 currentIndices[currentDim] == 0
    #                 return
    #             self.recursiveDis(points, currentIndices, currentDim - 1, resultVectors)
    #             currentIndices[currentDim] += 1
        

    # Calculate the RBFnets output for a given Input
    def calc(self, inputVector):
        # Calculate RBFLayer output
        rbfOutputs = np.zeros(self.rbfSize)
        for i, x in enumerate(self.rbfNeurons):
            rbfOutputs[i] = x.activate(inputVector)

        # Calculate final output
        outputs = np.zeros(self.outputSize)
        for i, x in enumerate(self.outputNeurons):
            outputs[i] = x.activate(rbfOutputs)
        return outputs, rbfOutputs

    # Train the network on a single pattern
    def train(self, pattern):
        inputXY = pattern[:2] # Input
        netOutput, rbfOutputs = self.calc(inputXY) # RBFNets output
        error =  pattern[2:] - netOutput
        for index, outputNeuron in enumerate(self.outputNeurons): # Train every Outputneuron
            delta_m = error[index]
            etha_delta_m = self.learnRate * delta_m
            change = etha_delta_m * rbfOutputs[index]
            outputNeuron.changeWeight(index,change)
    
    # Evaluate the network based on a single pattern
    def evaluate(self, pattern):
        inputXY = pattern[:2]
        netOutput, rbfOutputs = self.calc(inputXY)
        error = pattern[2:] - netOutput
        return error



class RBFNeuron:
    def __init__(self, center, width) -> None:
        self.center = center
        self.width = width
        
    # Calculate the RBFNeurons output based on an input
    def activate(self, input):
        distance = np.linalg.norm(input-self.center)
        return gaussian(distance, self.width)

class OutputNeuron:
    def __init__(self, weights) -> None:
        self.weights = weights

    # Calculate the OutputNeurons output based on the outputs of the RBFLayer
    def activate(self, inputs):
        return np.sum(self.weights * inputs)

    def changeWeight(self, weight, change):
        self.weights[weight] += change
    

def gaussian(x, s):
    return math.exp((x * x) / (-2 * s * s))