import numpy as np
import math

class RBFNet:

    def __init__(self, inputSize, rbfSize, outputSize) -> None:
        self.rbfSize = rbfSize
        self.outputSize = self.outputSize

        self.rbfNeurons = []
        for i in range(rbfSize):
            self.rbfNeurons.append(RBFNeuron(np.random.uniform(low=0, high=1.0, size=inputSize), np.random.uniform(0, 1.0, 1)))

        self.outputNeurons = []
        for i in range(outputSize):
            self.outputNeurons.append(OutputNeuron(np.random.uniform(low=-0.5, high=0.5, size=rbfSize)))
        

    def calc(self, inputVector):
        rbfOutputs = np.zeros(self.rbfSize)
        for i, x in enumerate(self.rbfNeurons):
            rbfOutputs[i] = x.activate(inputVector)

        outputs = np.zeros(self.outputSize)
        for i, x in enumerate(self.outputNeurons):
            outputs[i] = x.activate(rbfOutputs)
        return outputs


class RBFNeuron:
    
    def __init__(self, center, width) -> None:
        self.center = center
        self.width = width
        

    def activate(self, input):
        distance = np.linalg.norm(input, self.center)
        return gaussian(distance, self.width)

class OutputNeuron:

    def __init__(self, weights) -> None:
        self.weights = weights

    def activate(self, inputs):
        return np.sum(self.weights * inputs)
    

def gaussian(x, s):
    math.exp((x * x) / (-2 * s * s))