from math import exp
from random import seed, random, shuffle
import operator


class Neuron:
    # Create neuron
    def __init__(self):
        self.delta = 0
        self.last_output = 0

    # 0 because of inheritance
    def get_output_val(self):
        return 0

    # clear preveus leurned delta
    def clear_delta(self):
        self.delta = 0


    # because of inheritance a child class will overwrite this section of code
    @staticmethod
    def derivative(output):
        return output * (1.0 - output)
    # because of inheritance
    def calculate_delta_for_inputs(self):
        pass
    # because of inheritance
    def update_weights(self, learnRate):
        pass
    # because of inheritance
    def get_weights(self):
        return []


class HiddenNeuron(Neuron):
    # Create Hidden neuron in network/ load weights in to the network
    def __init__(self, inputs, is_loading=False):
        super().__init__()
        if is_loading:
            self.inputs = []
            self.bias_weight = inputs[-1][0]
            for w in inputs[:-1]:
                if len(w[1]) == 0:
                    self.inputs.append([w[0], InputNeuron()])
                else:
                    self.inputs.append([w[0], HiddenNeuron(w[1], True)])
        else:
            self.bias_weight = random()
            if issubclass(type(inputs), Neuron):
                weight = 0
                self.inputs = [[weight, inputs]]
            elif type(inputs) is list:
                if issubclass(type(inputs[0]), Neuron):
                    self.inputs = self.gen_weights(inputs)
                elif type(inputs[0]) is list:
                    if issubclass(type(inputs[0][1]), Neuron):
                        self.inputs = inputs
    # Returns activation of neuron
    def get_output_val(self):
        val = 0
        for neuron in self.inputs:
            val += neuron[0] * neuron[1].get_output_val()
        val += self.bias_weight
        # self.last_output = max(0, val)
        self.last_output = 1.0 / (1.0 + exp(-val))
        return self.last_output

    @staticmethod
    # Create random weights.
    def gen_weights(inputs):
        newinputs = []
        for i in inputs:
            weight = random()
            newinputs.append([weight, i])
        return newinputs

    # Clear the learned delta.
    def clear_delta(self):
        super().clear_delta()
        for i in self.inputs:
            i[1].clear_delta()

    # Calc delta
    def calculate_delta_for_inputs(self):
        for neuron in self.inputs:
            neuron[1].delta += (neuron[0] * self.delta) * self.derivative(neuron[1].last_output)
            neuron[1].calculate_delta_for_inputs()
    # Updates the current weights based on given delta.
    def update_weights(self, learnRate):
        for i in self.inputs:
            i[1].update_weights(learnRate)
            i[0] += learnRate * self.delta * i[1].last_output
        self.bias_weight += learnRate * self.delta

    # Return the weights (for saving the network).
    def get_weights(self):
        w = []
        for i in self.inputs:
            w.append([i[0], i[1].get_weights()])
        w.append([self.bias_weight, []])
        return w
    # Return the inputs (for loading the network).
    def get_inputs(self):
        if type(self.inputs[0][1]) == InputNeuron:
            inputs = []
            for i in self.inputs:
                inputs.append(i[1])
            return inputs
        return self.inputs[0][1].get_inputs()


class InputNeuron(Neuron):
    # Creates the nueron
    def __init__(self):
        super().__init__()
        self.val = 0
    # returns the value of the input neuron.
    def get_output_val(self):
        return self.val
    # Sets the value of the input neuron.
    def set_val(self, val):
        self.last_output = val
        self.val = val


class NN:
    # Creates the network voor a given neurons 'n'.
    def __init__(self, n=None):
        if n is None:
            n = []
        if len(n) > 1:  #checks if there is an in/out-put
            seed(1)
            self.inputs = []
            for i in range(n[0]):
                self.inputs.append(InputNeuron())

            self.outputs = self.inputs.copy()
            for i in n[1:]:
                self.add_layer(i)
        else:
            self.inputs = []
            self.outputs = []

    # Adds a layer with 'n' neurons to the network.
    def add_layer(self, n):
        newoutput = []
        for i in range(n):
            newoutput.append(HiddenNeuron(self.outputs))
        self.outputs = newoutput

    # Propegates back over the network and changes the weigths to be more fitting to the excptect output.
    def backward_propagation(self, expected):
        for output in self.outputs:
            output.clear_delta()

        for j in range(len(self.outputs)):
            neuron = self.outputs[j]
            error = (expected[j] - neuron.last_output)
            neuron.delta = error * neuron.derivative(neuron.last_output)

        for output in self.outputs:
            output.calculate_delta_for_inputs()

    # data_set: set of data to use as inputs
    # learn_set: set of data of the expected result of the network
    # learn_rate:
    # n_epoch: number of times to loop
    # print_errors: print the amount of error per epoch
    def train(self, data_set, learn_set, learn_rate=1.0, n_epoch=50, print_errors=False):
        for epoch in range(n_epoch):
            combined = list(zip(data_set, learn_set))
            shuffle(combined)
            data_set[:], learn_set[:] = zip(*combined)
            error = 0
            for t in range(len(data_set)):
                outputs = self.run(data_set[t])
                for i in range(len(outputs)):
                    error += (learn_set[t][i] - outputs[i]) ** 2
                self.backward_propagation(learn_set[t])
                for o in self.outputs:
                    o.update_weights(learn_rate)
            if print_errors:
                print('>epoch=%d, learnRate=%.3f, error=%.3f' % (epoch, learn_rate, error))

    # Runs the algorithme and returns the output of the network
    def run(self, input):
        if len(input) is not len(self.inputs):
            return None
        for i in range(len(input)):
            self.inputs[i].set_val(input[i])

        out = []
        for o in self.outputs:
            out.append(o.get_output_val())
        return out

    # Tests the network on bases of the givven data set. Will return a value of centriy of corretness and the resulting correctness.
    def test(self, dataSet, learnSet, print_result=False):
        combined = list(zip(dataSet, learnSet))
        shuffle(combined)
        dataSet[:], learnSet[:] = zip(*combined)
        averagep = 0
        correct = 0
        for i in range(len(dataSet)):
            out = self.run(dataSet[i])
            p = 0
            for j in range(len(out)):
                if learnSet[i][j] == 1:
                    p += out[j]
                if learnSet[i][j] == 0:
                    p -= out[j]
            index, value = max(enumerate(out), key=operator.itemgetter(1))
            if learnSet[i][index] == 1:
                correct += 1
            averagep += p * 100
            if print_result:
                print("Test: ", end='')
                print(learnSet[i], end='\t')
                print("Output: ", end='')
                print(out, end='\t')
                print(round(p * 100, 3), end='%\n')
        return [round(averagep / len(dataSet), 3), (correct / len(dataSet)) * 100]

    # Saves all the weights to list and returns this list
    def save(self):
        w = []
        for o in self.outputs:
            w.append(o.get_weights())
        return w

    #loads neural network (with the givven weigths)
    def load(self, weights):
        for w in weights:
            self.outputs.append(HiddenNeuron(w, True))
        self.inputs = self.outputs[0].get_inputs()

# If this isn't a import
if __name__ == "__main__":
    # then import numpy
    import numpy as numpy

    # Make classifactions
    def converter_type(s):
        s = s.decode("utf-8")
        r = [0] * 3
        if s == "Iris-setosa":
            r[0] = 1
        elif s == "Iris-versicolor":
            r[1] = 1
        elif s == "Iris-virginica":
            r[2] = 1
        return r

    # Set datasets
    dataSet = numpy.genfromtxt("flowers.csv", delimiter=",", usecols=[0, 1, 2, 3], converters={})
    learnSet = numpy.genfromtxt("flowers.csv", delimiter=",", usecols=[4], converters={4: converter_type})

    # Shuffel data set.
    combined = list(zip(dataSet, learnSet))
    shuffle(combined)
    shuffle(combined)
    shuffle(combined)
    shuffle(combined)
    dataSet[:], learnSet[:] = zip(*combined)

    # Run this b*
    nn = NN([len(dataSet[0]), 5, len(learnSet[0])])
    print("Total accuracy before learn: ", end='')
    print(nn.test(dataSet[round(len(dataSet) / 3):], learnSet[round(len(dataSet) / 3):], False), end='%\n')
    nn.train(dataSet[:round(len(dataSet) / 3)], learnSet[:round(len(dataSet) / 3)], 0.1, 300, False)
    print("Total accuracy after learn: ", end='')
    print(nn.test(dataSet[round(len(dataSet) / 3):], learnSet[round(len(dataSet) / 3):], False), end='%\n')

    nn2 = NN()
    nn2.load(nn.save())
    print("Total accuracy after load: ", end='')
    print(nn2.test(dataSet[round(len(dataSet) / 3):], learnSet[round(len(dataSet) / 3):], False), end='%\n')