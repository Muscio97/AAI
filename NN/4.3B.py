import random

class Neuron:
    def __init__(self):
        self.delta = 0
        self.last_output = 0

    def get_output_val(self):
        return 0

    def clear_delta(self):
        self.delta = 0

    @staticmethod
    def derivative(output):
        return output * (1.0 - output)

    def calculate_delta_for_inputs(self):
        pass

    def update_weights(self, learnRate):
        pass


class HiddenNeuron(Neuron):
    def __init__(self, inputs):
        super().__init__()
        self.bias = InputNeuron()
        self.bias.set_val(-1)
        self.bias_weight = random.uniform(-1, 1)
        if issubclass(type(inputs), Neuron):
            weight = 0
            self.inputs = [[weight, inputs]]
        elif type(inputs) is list:
            if issubclass(type(inputs[0]), Neuron):
                self.inputs = self.gen_weights(inputs)
            elif type(inputs[0]) is list:
                if issubclass(type(inputs[0][1]), Neuron):
                    self.inputs = inputs

    def get_output_val(self):
        val = 0
        for neuron in self.inputs:
            val += neuron[0] * neuron[1].get_output_val()
        val += self.bias_weight * self.bias.get_output_val()
        self.last_output = max(0, val)
        return self.last_output

    @staticmethod
    def gen_weights(inputs):
        newinputs = []
        for i in inputs:
            weight = random.uniform(-1, 1)
            newinputs.append([weight, i])
        return newinputs

    def clear_delta(self):
        super().clear_delta()
        for i in self.inputs:
            i[1].clear_delta()

    def calculate_delta_for_inputs(self):
        for neuron in self.inputs:
            neuron[1].delta += (neuron[0] * self.delta) * self.derivative(neuron[1].last_output)
            neuron[1].calculate_delta_for_inputs()

    def update_weights(self, learnRate):
        for i in self.inputs:
            i[1].update_weights(learnRate)
            i[0] += learnRate * self.delta * i[1].last_output
        self.bias_weight += learnRate * self.delta


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.val = 0

    def get_output_val(self):
        return self.val

    def set_val(self, val):
        self.last_output = val
        self.val = val


class NN:
    def __init__(self, n):
        self.inputs = []
        for i in range(n[0]):
            self.inputs.append(InputNeuron())

        self.outputs = self.inputs.copy()
        for i in n[1:]:
            self.add_layer(i)

    def add_layer(self, n):
        newoutput = []
        for i in range(n):
            newoutput.append(HiddenNeuron(self.outputs))
        self.outputs = newoutput

    def backward_propagation(self, expected):
        for output in self.outputs:
            output.clear_delta()

        for j in range(len(self.outputs)):
            neuron = self.outputs[j]
            error = (expected[j] - neuron.last_output)
            neuron.delta = error * neuron.derivative(neuron.last_output)

        for output in self.outputs:
            output.calculate_delta_for_inputs()

    def train(self, trainSet, learnRate, ntimes, expectedSet):
        for e in range(ntimes):
            error = 0
            for t in range(len(trainSet)):
                outputs = self.run(trainSet[t])
                expected = expectedSet[t]
                for i in range(len(outputs)):
                    error += (expected[i] - outputs[i]) ** 2
                self.backward_propagation(expected)
                for o in self.outputs:
                    o.update_weights(learnRate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (e, learnRate, error))

    def run(self, input):
        if len(input) is not len(self.inputs):
            return None
        for i in range(len(input)):
            self.inputs[i].set_val(input[i])

        out = []
        for o in self.outputs:
            out.append(o.get_output_val())
        return out


dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
testset = []
for d in range(len(dataset)):
    expected = [0 for i in range(n_outputs)]
    expected[dataset[d][-1]] = 1
    testset.append(expected)
    dataset[d] = dataset[d][:-1]

nn = NN([n_inputs, 5, n_outputs])
print(testset[5])
print(nn.run(testset[5]))
nn.train(dataset, 1, 50, testset)
for i in range(5):
    print()
    print(testset[i])
    print(nn.run(testset[i]))
