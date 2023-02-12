from abc import ABC
import random
import math


class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, num_hidden_layer_neurons):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs


class Neuron(ABC):
    def __init__(self, name):
        self._name = name
        self._activation_function = None
        self._neuron_type = None

    def get_name(self):
        return self._name

    def calc_activation_function(self, value):
        if self._activation_function == "sigmoid":
            return 1/(1+math.e**(-value))

        elif self._activation_function == "tanh":
            return 2/(1+math.e**(-2*value))-1

        elif self._activation_function == "relu":
            return max(0, value)

        elif self._activation_function == "identity":
            return value

        elif self._activation_function == "binary":
            return 1 if value >= 1 else 0

        else:
            print("(!) No activation function set")

    def __str__(self):
        return f"\n[{self._neuron_type} Neuron] {self._name}"


class InputNeuron(Neuron):
    def __init__(self, name):
        super().__init__(name)
        self._input_value = 0
        self._neuron_type = "Input"

    def set_input_value(self, value):
        self._input_value = value

    def get_input_value(self):
        return self._input_value

    def __str__(self):
        return_str = super().__str__()
        return_str += f"\n  Current Input Value: {self._input_value}"

        return return_str


class HiddenNeuron(Neuron):
    def __init__(self, name, input_neuron_list, activation_function):
        super().__init__(name)

        self._weights = {name: round(random.random(), 4) for name in input_neuron_list}
        self._bias = 1
        self._activation_function = activation_function
        self._neuron_type = "Hidden"

    def get_weights(self):
        return self._weights

    def calc_output(self, input_dict):
        sum_of_inputs = 0

        # Sum up neurons * weights
        for neuron in input_dict:
            sum_of_inputs += input_dict[neuron] * self._weights[neuron]

        # Add bias
        sum_of_inputs += self._bias

        # Activation function
        return self.calc_activation_function(sum_of_inputs)

    def __str__(self):
        return_str = super().__str__()
        return_str += f"\n  Activation: {self._activation_function}"
        return_str += f"\n  Bias: {self._bias}"
        return_str += f"\n  Weights: {self._weights}"

        return return_str


class OutputNeuron(Neuron):
    def __init__(self, name, input_neuron_list, activation_function):
        super().__init__(name)

        self._weights = {name: round(random.random(), 4) for name in input_neuron_list}
        self._bias = 1
        self._activation_function = activation_function
        self._neuron_type = "Output"

    def get_weights(self):
        return self._weights

    def calc_output(self, input_dict):
        sum_of_inputs = 0

        # Sum up neurons * weights
        for neuron in input_dict:
            sum_of_inputs += input_dict[neuron] * self._weights[neuron]

        # Add bias
        sum_of_inputs += self._bias

        # Activation function
        return self.calc_activation_function(sum_of_inputs)

    def __str__(self):
        return_str = super().__str__()
        return_str += f"\n  Activation: {self._activation_function}"
        return_str += f"\n  Bias: {self._bias}"
        return_str += f"\n  Weights: {self._weights}"

        return return_str


if __name__ == "__main__":
    h1 = HiddenNeuron("Test", ["h1", "h2", "h3"], "tanh")
    print(h1.get_weights())
    print(h1.calc_activation_function(-2))
    print(h1)
