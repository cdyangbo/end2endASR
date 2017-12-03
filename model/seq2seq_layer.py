#-*- coding:utf-8 -*-
import os,time,datetime
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod


def pyramid_stack(inputs, sequence_lengths, scope=None, time_major=False):
    '''
    concatenate each two consecutive elements

    Args:
        inputs: A time minor tensor [batch_size,time,input_size], else [time, batch_size, input_size]
        sequence_lengths: the length of the input sequences
        scope: the current scope

    Returns:
        inputs: Concatenated inputs time minor [batch_size, time/2, input_size*2],else [ time/2,,batch_size,input_size*2]
        sequence_lengths: the lengths of the inputs sequences [batch_size]
    '''

    with tf.name_scope(scope or 'pyramid_stack'):

        input_shape = tf.Tensor.get_shape(inputs)

        if time_major:
            # pad with zeros if odd number of inputs
            if int(input_shape[0]) % 2 == 1:
                padded_inputs = tf.pad(inputs, [[0, 1], [0, 0], [0, 0]])
                length = int(input_shape[0]) + 1
            else:
                padded_inputs = inputs
                length = int(input_shape[0])

            # convert imputs to time major
            # time_major_input = tf.transpose(padded_inputs, [1, 0, 2])

            # seperate odd and even inputs
            odd_inputs = tf.gather(padded_inputs, range(1, length, 2))
            even_inputs = tf.gather(padded_inputs, range(0, length, 2))

            # concatenate odd and even inputs
            outputs = tf.concat([even_inputs, odd_inputs], 2)

            # convert back to time minor
            # outputs = tf.transpose(time_major_outputs, [1, 0, 2])
        else:
            # pad with zeros if odd number of inputs
            if int(input_shape[1]) % 2 == 1:
                padded_inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 0]])
                length = int(input_shape[1]) + 1
            else:
                padded_inputs = inputs
                length = int(input_shape[1])

            # convert imputs to time major
            time_major_input = tf.transpose(padded_inputs, [1, 0, 2])

            # seperate odd and even inputs
            odd_inputs = tf.gather(time_major_input, range(1, length, 2))
            even_inputs = tf.gather(time_major_input, range(0, length, 2))

            # concatenate odd and even inputs
            time_major_outputs = tf.concat([even_inputs, odd_inputs], 2)

            # convert back to time minor
            outputs = tf.transpose(time_major_outputs, [1, 0, 2])

        # compute the new sequence length
        output_sequence_lengths = tf.cast(tf.ceil(tf.cast(sequence_lengths, tf.float32) / 2), tf.int32)

    return outputs, output_sequence_lengths

def nonseq2seq(tensor, seq_length, length, name=None, time_major=False):
    '''
    Convert non sequential data to sequential data

    Args:
        tensor: non sequential data, which is a TxF tensor where T is the sum of
            all sequence lengths
        seq_length: a vector containing the sequence lengths
        length: the constant length of the output sequences
        name: [optional] the name of the operation

    Returns:
        sequential data, which is a [batch_size, max_length, dim]
        tensor
    '''

    with tf.name_scope(name or'nonseq2seq'):
        #get the cumulated sequence lengths to specify the positions in tensor
        cum_seq_length = tf.concat([tf.constant([0]), tf.cumsum(seq_length)], 0)

        #get the indices in the tensor for each sequence
        indices = [tf.range(cum_seq_length[l], cum_seq_length[l+1])
                   for l in range(int(seq_length.get_shape()[0]))]

        #create the non-padded sequences
        sequences = [tf.gather(tensor, i) for i in indices]

        #pad the sequences with zeros
        sequences = [tf.pad(sequences[s], [[0, length-seq_length[s]], [0, 0]])
                     for s in range(len(sequences))]

        #specify that the sequences have been padded to the constant length
        for seq in sequences:
            seq.set_shape([length, int(tensor.get_shape()[1])])

        #stack the sequences into a tensor
        if time_major:
            sequential = tf.stack(sequences,axis=1)
        else:
            sequential = tf.stack(sequences,axis=0)
    return sequential

def seq2nonseq(sequential, seq_length, name=None,time_major = False):
    '''
    Convert sequential data to non sequential data

    Args:
        sequential: the sequential data which is a [batch_size, max_length, dim]
            tensor if time major=false,else [t,b,f]
        seq_length: a vector containing the sequence lengths
        name: [optional] the name of the operation

    Returns:
        non sequential data, which is a TxF tensor where T is the sum of all
        sequence lengths,T=txb or bxt
    '''

    with tf.name_scope(name or 'seq2nonseq'):
        #convert the list for each time step to a list for each sequence
        if time_major:
            sequences = tf.unstack(sequential, axis = 1)
        else:
            sequences = tf.unstack(sequential, axis = 0)

        #remove the padding from sequences
        sequences = [tf.gather(sequences[s], tf.range(seq_length[s]))
                     for s in range(len(sequences))]

        #concatenate the sequences
        tensor = tf.concat(sequences, 0)

    return tensor


class BRNNLayer(object):
    """This class allows enables RNN layer creation as well as computing
       their output. The output is found by linearly combining the forward
       and backward pass as described in:
       Graves et al., Speech recognition with deep recurrent neural networks,
       page 6646.
    """

    def __init__(self, num_units, cell_fn=tf.contrib.rnn.BasicLSTMCell, activation=tf.nn.relu):
        '''

        :param num_units:
        :param cell_fn:
        :param activation:
        '''

        self.num_units = num_units
        self.cell_fn = cell_fn
        self.activation = activation

    def __call__(self, inputs, sequence_length, scope=None, time_major=False):
        """
        Create the variables and do the forward computation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor if time_major=false,
            sequence_length: the length of the input sequences
            scope: The variable scope sets the namespace under which
                      the variables created during this call will be stored.

        Returns:
            the output of the layer
        """

        with tf.variable_scope(scope or type(self).__name__):
            # create the lstm cell that will be used for the forward and backward
            # pass
            rnn_cell_fw = self.cell_fn(self.num_units,
                                       #reuse=tf.get_variable_scope().reuse,
                                       activation=self.activation)
            rnn_cell_bw = self.cell_fn(self.num_units,
                                       #reuse=tf.get_variable_scope().reuse,
                                       activation=self.activation)

            # do the forward computation
            outputs_tupple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fw,
                                                                cell_bw=rnn_cell_bw,
                                                                inputs=inputs,
                                                                dtype=tf.float32,
                                                                sequence_length=sequence_length,
                                                                time_major=time_major)

            outputs = tf.concat(outputs_tupple, 2)

            return outputs


class PydBRNNLayer(object):
    ''' a pyramidal bidirectional RNN layer'''

    def __init__(self, num_units, cell_fn=tf.contrib.rnn.BasicLSTMCell, activation=tf.nn.relu):
        """
        BlstmLayer constructor
        Args:
            num_units: The number of units in the LSTM
            pyramidal: indicates if a pyramidal BLSTM is desired.
        """

        # create Brnn layer
        self.brnn = BRNNLayer(num_units, cell_fn, activation)

    def __call__(self, inputs, sequence_lengths, scope=None, time_major=False):
        """
        Create the variables and do the forward computation
        Args:
            inputs: A time minor tensor of shape [batch_size, time,
                input_size],else time_major
            sequence_lengths: the length of the input sequences
            scope: The variable scope sets the namespace under which
                the variables created during this call will be stored.
        Returns:
            the output of the layer, the concatenated outputs of the
            forward and backward pass shape [batch_size, time/2, input_size*2].
        """

        with tf.variable_scope(scope or type(self).__name__):
            # apply blstm layer
            outputs = self.brnn(inputs, sequence_lengths, time_major=time_major)
            stacked_outputs, output_seq_lengths = pyramid_stack(outputs, sequence_lengths, time_major=time_major)
        return stacked_outputs, output_seq_lengths

class Encoder(object):
    '''
    a general encoder object transforms input features into a high level representation
    '''

    __metaclass__ = ABCMeta

    def __init__(self, conf, name=None):
        '''
        Encoder constructor
        '''
        #save the parameters
        self.conf = conf

        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, sequence_lengths, is_training, time_major=False):
        '''
        Create the variables and do the forward computation
        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor if time_major=false, else [max_length,batch_size, dim]
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode
        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''

        with tf.variable_scope(self.scope):
            outputs = self.encode(inputs, sequence_lengths, is_training, time_major=time_major)
        self.scope.reuse_variables()

        return outputs

    @abstractmethod
    def encode(self, inputs, sequence_lengths, is_training, time_major=False):
        '''
        get the high level feature representation

        Args:
            inputs: the input to the layer as a
                [batch_size, max_length, dim] tensor if time_major = false ,otherwise exchane dim0,dim1
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a [bath_size, max_length, output_dim]
            tensor
        '''


class Listener(Encoder):
    '''
    a listener object transforms input features into a high level representation
    '''

    def __init__(self, conf, name=None):
        '''
        Listener constructor
        Args:
            numlayers: the number of PBLSTM layers
            numunits: the number of units in each layer
            dropout: the dropout rate
            name: the name of the Listener
        '''

        self.numlayers = conf['listener_numlayers']
        self.numunits = conf['listener_numunits']
        self.dropout = conf['listener_dropout']

        if conf['listener_cell'] == 'gru':
            self.cell_fn = tf.contrib.rnn.GRUCell
        elif conf['listener_cell'] == 'lstm':
            self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif conf['listener_cell'] == 'rnn':
            self.cell_fn = tf.contrib.rnn.BasicRNNCell
        else:
            raise NotImplementedError(conf['listener_cell'])

        if conf['listener_activation'] == 'relu':
            self.activation = tf.nn.relu
        else:  # default
            self.activation = tf.nn.tanh

        # create the pblstm layer
        self.pbrnn = PydBRNNLayer(num_units=self.numunits, cell_fn=self.cell_fn, activation=self.activation)

        # create the blstm layer
        self.brnn = BRNNLayer(num_units=self.numunits, cell_fn=self.cell_fn, activation=self.activation)

        super(Listener, self).__init__(conf, name)


    def encode(self, inputs, sequence_lengths, is_training=False, time_major=False):
        '''
        get the high level feature representation

        Args:
            inputs: the input to the layer as a time minor [batch_size, max_length, dim] or time major,tensor
            sequence_length: the length of the input sequences
            is_training: whether or not the network is in training mode

        Returns:
            the output of the layer as a time minor [bath_size, max_length, output_dim] or time major
            tensor
        '''

        outputs = inputs
        output_seq_lengths = sequence_lengths
        for l in range(int(self.numlayers)):
            outputs, output_seq_lengths = self.pbrnn(outputs, output_seq_lengths, 'Listener%d' % l, time_major=time_major)

            if float(self.conf['listener_dropout']) < 1 and is_training:
                outputs = tf.nn.dropout(outputs, float(self.conf['listener_dropout']))

        outputs = self.brnn(outputs, output_seq_lengths, 'Listener%d' % int(self.numlayers), time_major=time_major)

        if float(self.dropout) < 1 and is_training:
            outputs = tf.nn.dropout(outputs, float(self.dropout))

        return outputs

class Decoder(object):
    '''a general seq2seq decoder object
    converts the high level features into output logits'''

    __metaclass__ = ABCMeta

    def __init__(self, conf, output_dim,name=None):
        '''Decoder constructor

        Args:
            conf: the classifier config as a dictionary
            output_dim: the classifier output dimension
            name: the speller name'''


        #save the parameters
        self.conf = conf
        self.output_dim = output_dim

        self.scope = tf.VariableScope(False, name or type(self).__name__)


    def __call__(self, hlfeat, encoder_inputs, initial_state, first_step,
                 is_training,time_major=False):
        '''
        Create the variables and do the forward computation

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim] if time major=false
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length] if time_major=false.
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor if time_major = false
            - the final state of the listener
        '''

        with tf.variable_scope(self.scope):

            logits, state = self.decode(hlfeat, encoder_inputs, initial_state,
                                        first_step, is_training, time_major=time_major)

        self.scope.reuse_variables()

        return logits, state

    @abstractmethod
    def decode(self, hlfeat, encoder_inputs, initial_state, first_step,
               is_training, time_major=False):
        '''
        Get the logits and the output state

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim] if time_major=false
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length] if time_major=false.
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor if time_major=false
            - the final state of the listener
        '''

    @abstractmethod
    def zero_state(self, batch_size):
        '''get the decoder zero state

        Returns:
            an rnn_cell zero state'''


class Speller(Decoder):
    '''a speller decoder for the LAS architecture'''

    def __init__(self, conf, output_dim, name=None):
        self.sample_prob = conf['speller_sample_prob']
        self.numlayers = conf['speller_numlayers']
        self.numunits = conf['speller_numunits']
        self.dropout = conf['speller_dropout']

        if conf['speller_cell'] == 'gru':
            self.cell_fn = tf.contrib.rnn.GRUCell
        elif conf['speller_cell'] == 'lstm':
            self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif conf['speller_cell'] == 'rnn':
            self.cell_fn = tf.contrib.rnn.BasicRNNCell
        else:
            raise NotImplementedError(conf['speller_cell'])

        if conf['speller_activation'] == 'relu':
            self.activation = tf.nn.relu
        else:  # default
            self.activation = tf.nn.tanh

        super(Speller,self).__init__(conf,output_dim,name)

    def decode(self, hlfeat, encoder_inputs, initial_state, first_step,
               is_training, time_major=False):
        '''
        Get the logits and the output state

        Args:
            hlfeat: the high level features of shape
                [batch_size x hl_seq_length x feat_dim] if time major=false
            encoder_inputs: the one-hot encoded training targets of shape
                [batch_size x target_seq_length].if time major=false
            initial_state: the initial decoder state, could be usefull for
                decoding
            first_step: bool that determines if this is the first step
            is_training: whether or not the network is in training mode

        Returns:
            - the output logits of the listener as a
                [batch_size x target_seq_length x numlabels] tensor if time major=false
            - the final state of the listener
        '''

        # one hot encode the targets
        # one hot -> encoder_inputs.shape x self.output_dim ,with axis = -1
        one_hot_inputs = tf.one_hot(encoder_inputs, self.output_dim, dtype=tf.float32)


        # put targets in time major
        time_major_inputs = tf.transpose(one_hot_inputs, [1, 0, 2])

        # convert targets to list
        input_list = tf.unstack(time_major_inputs)

        # create the rnn cell
        rnn_cell = self.create_rnn(is_training)

        # create the loop functions
        lf = partial(loop_function, time_major_inputs, float(self.sample_prob))

        if time_major:
            # to [batch_size,att_length,att_size]
            hlfeat = tf.transpose(hlfeat,[1,0,2])

        # use the attention decoder
        logit_list, state = tf.contrib.legacy_seq2seq.attention_decoder(
            decoder_inputs=input_list,
            initial_state=initial_state,
            attention_states=hlfeat,
            cell=rnn_cell,
            output_size=self.output_dim,
            loop_function=lf,
            scope='attention_decoder',
            initial_state_attention=not first_step)

        logits = tf.stack(logit_list)

        if not time_major:
            logits = tf.transpose(logits, [1, 0, 2])

        return logits, state

    def create_rnn(self, is_training=False):
        '''created the decoder rnn cell

        Args:
            is_training: whether or not the network is in training mode

        Returns:
            an rnn cell'''

        rnn_cells = []

        for _ in range(int(self.numlayers)):
            # create the multilayered rnn cell
            rnn_cell = self.cell_fn(self.numunits,
                                    #reuse=tf.get_variable_scope().reuse,
                                    activation=self.activation)

            if float(self.dropout) < 1 and is_training:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell,
                                                         output_keep_prob=float(self.dropout))

            rnn_cells.append(rnn_cell)

        rnn_cell = tf.contrib.rnn.MultiRNNCell(rnn_cells)

        return rnn_cell

    def zero_state(self, batch_size):
        '''get the listener zero state

        Returns:
            an rnn_cell zero state'''

        return self.create_rnn().zero_state(batch_size, tf.float32)


def loop_function(decoder_inputs, sample_prob, prev, i):
    '''the loop function used in the attention decoder_inputs, used for
    scheduled sampling

    Args:
        decoder_inputs: the ground truth labels as a tensor of shape
            [seq_length, batch_size, numlabels] (time_major)
        sample_prob: the probability that the network will sample the output
        prev: the outputs of the previous steps
        i: the current decoding step

    returns:
        the input for the nect time step
    '''

    batch_size = int(decoder_inputs.get_shape()[1])
    numlabels = decoder_inputs.get_shape()[2]

    # get the most likely characters as the sampled output
    next_input_sampled = tf.one_hot(tf.argmax(prev, 1), numlabels)

    # get the current ground truth labels
    next_input_truth = tf.gather(decoder_inputs, i)

    # creat a boolean vector of where to sample
    sample = tf.less(tf.random_uniform([batch_size]), sample_prob)

    next_input = tf.where(sample, next_input_sampled, next_input_truth)

    return next_input


class TSClassifier(object):
    '''This an abstract class defining a neural net time series classifier'''
    __metaclass__ = ABCMeta
    def __init__(self, conf, output_dim, name=None):
        '''classifier constructor
        Args:
            conf: The classifier configuration
            output_dim: the classifier output dimension
            name: the classifier name
        '''

        self.conf = conf
        self.output_dim = int(output_dim)
        self.input_dim = conf['input_dim']

        #increase the output dim with the amount of labels that should be added
        self.output_dim += int(conf['add_labels'])

        #create the variable scope for the classifier
        self.scope = tf.VariableScope(False, name or type(self).__name__)

    def __call__(self, inputs, input_seq_length, targets,
                 target_seq_length, is_training, time_major=False):
        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor for time minor,other time_major
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode
            time_major: if true ,first dim of tensor is time

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        with tf.variable_scope(self.scope):
            outputs, output_seq_lengths = self._get_outputs(
                inputs, input_seq_length, targets, target_seq_length,
                is_training,time_major=time_major)

        #put the reuse flag to true in the scope to make sure the variables are
        #reused in the next call
        self.scope.reuse_variables()

        return outputs, output_seq_lengths

    @abstractmethod
    def _get_outputs(self, inputs, input_seq_length, targets,
                     target_seq_length, is_training,time_major=False):

        '''
        Add the neural net variables and operations to the graph

        Args:
            inputs: the inputs to the neural network, this is a
                [batch_size x max_input_length x feature_dim] tensor for time minor
            input_seq_length: The sequence lengths of the input utterances, this
                is a [batch_size] vector
            targets: the targets to the neural network, this is a
                [batch_size x max_output_length] tensor. The targets can be
                used during training
            target_seq_length: The sequence lengths of the target utterances,
                this is a [batch_size] vector
            is_training: whether or not the network is in training mode

        Returns:
            A pair containing:
                - output logits
                - the output logits sequence lengths as a vector
        '''

        raise NotImplementedError("Abstract method")

    @property
    def variables(self):
        '''get all variables from this model
        Returns:
            a list of model variables
        '''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=self.scope.name)

    def compute_ctc_loss(self, targets, logits, logit_seq_length,
                         target_seq_length, time_major=False):
        '''
        Compute the loss

        Creates the operation to compute the CTC loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length] tensor containing the
                targets
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits for time minor
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                [batch_size] vector
            time_major:

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('CTC_loss'):
            #get the batch size
            targets = tf.expand_dims(targets, 2)
            batch_size = int(targets.get_shape()[0])

            #convert the targets into a sparse tensor representation
            #为什么不直接用sparsetensor呢？
            indices = tf.concat([tf.concat(
                [tf.expand_dims(tf.tile([s], [target_seq_length[s]]), 1),
                 tf.expand_dims(tf.range(target_seq_length[s]), 1)], 1)
                                 for s in range(batch_size)], 0)

            values = tf.reshape(seq2nonseq(targets, target_seq_length,time_major=time_major), [-1])

            shape = [batch_size, int(targets.get_shape()[1])]

            sparse_targets = tf.SparseTensor(tf.cast(indices, tf.int64), values, shape)

            loss = tf.reduce_mean(tf.nn.ctc_loss(sparse_targets, logits,
                                                 logit_seq_length,
                                                 time_major=time_major))

        return loss

    def compute_ce_loss(self, targets, logits, logit_seq_length,
                     target_seq_length,time_major=False):
        '''
        Compute the loss

        Creates the operation to compute the cross-enthropy loss for every input
        frame (if you want to have a different loss function, overwrite this
        method)

        Args:
            targets: a [batch_size, max_target_length] tensor containing the
                targets
            logits: a [batch_size, max_logit_length, dim] tensor containing the
                logits if time minor
            logit_seq_length: the length of all the logit sequences as a
                [batch_size] vector
            target_seq_length: the length of all the target sequences as a
                [batch_size] vector
            time_major:

        Returns:
            a scalar value containing the loss
        '''

        with tf.name_scope('cross_enthropy_loss'):
            output_dim = int(logits.get_shape()[2]) # 字典的大小

            # put all the tragets on top of each other
            split_targets = tf.unstack(targets)
            for i, target in enumerate(split_targets):
                # only use the real data
                split_targets[i] = target[:target_seq_length[i]]

                # append an end of sequence label
                # output_dim -1 is <EOS>?
                split_targets[i] = tf.concat([split_targets[i], [output_dim-1]], 0)

            # concatenate the targets
            nonseq_targets = tf.concat(split_targets, 0)

            # convert the logits to non sequential data
            nonseq_logits = seq2nonseq(logits, logit_seq_length)

            # one hot encode the targets
            # pylint: disable=E1101
            nonseq_targets = tf.one_hot(nonseq_targets, output_dim)

            # compute the cross-enthropy loss
            # ? softmax_cross_entropy_with_logits should use one_hot targets?,
            # tf.losses.softmax_cross_entropy use one_hot
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=nonseq_logits,
                                                                          labels=nonseq_targets))

        return loss

    @abstractmethod
    def compute_conv_output_length(self, input_length, padding='SAME', stride =1):
        '''

        :param input_length:
        :param padding:
        :param stride:
        :return:
        '''