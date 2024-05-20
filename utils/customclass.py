from tensorflow.keras.layers import Layer
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import multiply


class NormalizeActivation(Layer):
    def __init__(self, epsilon=1e-10, **kwargs):
        super(NormalizeActivation, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        return inputs / (tf.reduce_sum(inputs, axis=-1, keepdims=True) + self.epsilon)

    def get_config(self):
        config = super(NormalizeActivation, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

class WeightedSumOfExperts(Layer):
    def __init__(self, **kwargs):
        super(WeightedSumOfExperts, self).__init__(**kwargs)

    def call(self, inputs):
        gating_weights = inputs[0]  # The gating mechanism's output
        experts_outputs = inputs[1:]  # The list of expert outputs

        # Multiply each expert's output by the corresponding gating weight
        weighted_experts = [multiply([gating_weights[:, i:i+1], expert_output])
                            for i, expert_output in enumerate(experts_outputs)]

        # Sum the weighted expert outputs
        return tf.add_n(weighted_experts)

    def get_config(self):
        # This method is needed for saving and loading the model
        config = super(WeightedSumOfExperts, self).get_config()
        return config


def add_normalize_activation(model):
    # Assuming the output of the model is the input to NormalizeActivation
    normalized_output = NormalizeActivation()(model.output)
    # Create a new model with the same input as the original and the new output
    new_model = Model(model.input, normalized_output)

    return new_model