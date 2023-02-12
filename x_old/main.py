from abc import ABC
from random import uniform

weights_range = [-2, 2]


class Neuron(ABC):
    def __init__(self, neuron_name):
        self._name = neuron_name
        self._activation_value = uniform(0,1)

    def get_name(self):
        return self._name

    def get_activation_value(self):
        return self._activation_value

    def set_activation_value(self, value):
        self._activation_value = value


class HiddenNeuron(Neuron):
    def __init__(self, neuron_name):
        super().__init__(neuron_name)
        self._connections = {}
        self._bias = 0

    def add_connection(self, origin_neuron, weight):
        self._connections[origin_neuron] = weight

    def get_all_connections(self):
        return self._connections

    def calculate_activation_value(self, layer_dict):
        val = 0
        for neuron in self._connections:
            val += layer_dict[neuron].get_activation_value() * self._connections[neuron]

        # Bias
        val += self._bias

        # Sigmoid function to create range from 0 to 1
        sigmoid = 1/(1+2.7182**-val)

        self.set_activation_value(sigmoid)


class InputNeuron(Neuron):
    def __init__(self, neuron_name):
        super().__init__(neuron_name)


class OutputNeuron(Neuron):
    def __init__(self, neuron_name):
        super().__init__(neuron_name)


class NeuralNetwork:
    def __init__(self, num_input_neurons, num_hidden_neurons, num_output_neurons):

        # Creating input neurons
        self._input_layer = {}

        counter = 1
        for i in range(num_input_neurons):
            neuron_name = "input_neuron_" + str(counter)
            self._input_layer[neuron_name] = (InputNeuron(neuron_name))
            counter += 1

        # Creating hidden neurons
        self._hidden_layer = {}

        counter = 1
        for i in range(num_hidden_neurons):
            neuron_name = "hidden_neuron_" + str(counter)
            self._hidden_layer[neuron_name] = (HiddenNeuron(neuron_name))
            counter += 1

        # Creating Connections and Weights
        for neuron in self._hidden_layer:
            for input_neuron in self._input_layer:
                self._hidden_layer[neuron].add_connection(input_neuron, round(uniform(weights_range[0], weights_range[1]), 2))

    def get_input_layer(self):
        return self._input_layer

    def get_hidden_layer(self):
        return self._hidden_layer

    def calculate_activation_values(self):
        for neuron in self._hidden_layer:
            self._hidden_layer[neuron].calculate_activation_value(self._input_layer)


n1 = NeuralNetwork(3, 5, 3)
n1.calculate_activation_values()

for i in n1.get_hidden_layer():
    print(n1.get_hidden_layer()[i].get_activation_value())
