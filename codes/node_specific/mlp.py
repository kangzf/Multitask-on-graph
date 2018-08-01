import tensorflow as tf
from layers import Dense
from link_specific.bilinear_diag import BilinearDiag
from additional.metrics import *


class MLP(BilinearDiag):
    def __init__(self, placeholders, settings, next_component=None, **kwargs):
        super(MLP, self).__init__(next_component, settings, **kwargs)

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.settings = settings
        self.next_component = next_component

        mode = kwargs.get('mode')
        # subject_codes, _, object_codes = self.next_component.get_all_codes(mode=mode)
        self.inputs = None # subject_codes + object_codes
        # e1s, _, e2s = self.compute_codes(mode='train')
        # self.input_dim = self.inputs.get_shape().as_list()[1]
        self.input_dim = int(self.settings['InternalEncoderDimension'])
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.layers = []
        self.activations = []
        self.outputs = None

        self.vars = {}
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        logging = kwargs.get('logging', False)
        self.logging = logging

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=int(self.settings['Algorithm']['learning_rate']))

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += float(self.settings['weight_decay']) * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        # return self.loss

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=round(self.input_dim / 4),
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=round(self.input_dim / 4),
                                 output_dim=round(self.input_dim / 16),
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 # sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=round(self.input_dim / 16),
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def get_loss(self, *args, default='loss'):
        """ Wrapper for _loss() and _accuracy()"""
        # self._loss()
        if default == 'loss':
            return self.loss
        elif default == 'acc':
            return self.accuracy
        else:
            raise ValueError

    def get_vars(self):
        return list(self.vars.values())

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

    def local_get_train_input_variables(self):
        return [self.placeholders['labels'],
                self.placeholders['labels_mask']]

        # self.opt_op = self.optimizer.minimize(self.loss)
    # def get_loss(self, mode='train'):
    #     e1s, rs, e2s = self.compute_codes(mode=mode)

    #     energies = tf.reduce_sum(e1s * rs * e2s, 1)

    #     weight = int(self.settings['NegativeSampleRate'])
    #     weight = 1
    #     return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.Y, energies, weight))

    # def predict(self):
    #     return tf.nn.softmax(self.outputs)
