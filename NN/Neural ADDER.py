class Neuron:
    @staticmethod
    def gen_weights(neurons):
        new_neurons = []
        for neuron in neurons:
            weight = 0
            new_neurons.append([weight, neuron])
        return new_neurons

    def get_output_val(self):
        return 0


class HiddenNeuron(Neuron):
    inputs = []

    def __init__(self, inputs):
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
        return 0 if val < 0 else 1


class InputNeuron(Neuron):
    val = 0

    def __init__(self):
        pass

    def get_output_val(self):
        return self.val

    def set_val(self, val):
        self.val = val


in1 = InputNeuron()
in2 = InputNeuron()
bias = InputNeuron()

bias.set_val(-1)
x11 = HiddenNeuron([[0.5, in1], [0.5, in2], [1, bias]])
x12 = HiddenNeuron([[-0.5, in1], [-0.5, in2], [0, bias]])
x13 = HiddenNeuron([[0.5, in1], [0.5, in2], [1, bias]])

out1 = HiddenNeuron([[-0.5, x11], [-0.5, x12], [0, x13], [0, bias]])
out2 = HiddenNeuron([[0, x11], [0, x12], [1, x13], [1, bias]])


def ADDER(a, b):
    in1.set_val(a)
    in2.set_val(b)
    return out1.get_output_val()


def ADDERC(a, b):
    in1.set_val(a)
    in2.set_val(b)
    return out2.get_output_val()


import itertools
from prettytable import PrettyTable
import re


class Gob(object):
    pass


class Truths(object):
    def __init__(self, base=None, phrases=None, ints=True):
        if not base:
            raise Exception('Base items are required')
        self.base = base
        self.phrases = phrases or []
        self.ints = ints

        # generate the sets of booleans for the bases
        self.base_conditions = list(itertools.product([False, True],
                                                      repeat=len(base)))

        # regex to match whole words defined in self.bases
        # used to add object context to variables in self.phrases
        self.p = re.compile(r'(?<!\w)(' + '|'.join(self.base) + ')(?!\w)')

    def calculate(self, *args):
        # store bases in an object context
        g = Gob()
        for a, b in zip(self.base, args):
            setattr(g, a, b)

        # add object context to any base variables in self.phrases
        # then evaluate each
        eval_phrases = []
        for item in self.phrases:
            item = self.p.sub(r'g.\1', item)
            eval_phrases.append(eval(item))

        # add the bases and evaluated phrases to create a single row
        row = [getattr(g, b) for b in self.base] + eval_phrases
        if self.ints:
            return [int(item) for item in row]
        else:
            return row

    def __str__(self):
        t = PrettyTable(self.base + self.phrases)
        for conditions_set in self.base_conditions:
            t.add_row(self.calculate(*conditions_set))
        return str(t)


print(Truths(['a', 'b'], ["ADDER(a, b)", "ADDERC(a, b)"]))
