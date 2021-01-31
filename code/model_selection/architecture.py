from widis_lstm_tools.nn import LSTMLayer, LearningRateDecay
from torch import nn
# Creating the model class using widis_lstm_tools library
class Net(nn.Module):
    def __init__(self, n_input_features, n_lstm, n_outputs, kernel_size, use_cnn, use_prefinal_layer):
        super(Net, self).__init__()
        # Let's say we want an LSTM with forward connections to cell input and recurrent connections to input- and
        # output gate only; Furthermore we want a linear LSTM output activation instead of tanh:
        self.lstm1 = LSTMLayer(
            in_features=n_input_features, out_features=n_lstm,
            # Possible input formats: 'NLC' (samples, length, channels), 'NCL', or 'LNC'
            inputformat='NLC',
            # cell input: initialize weights to forward inputs with xavier, disable connections to recurrent inputs
            # input gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            # output gate: disable connections to forward inputs, initialize weights to recurrent inputs with xavier
            # forget gate: disable all connection (=no forget gate) and bias
            w_fg=False, b_fg=False,
            # LSTM output activation shall be identity function
            # Optionally use negative input gate bias for long sequences
            b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),
            # Optionally let LSTM do computations after sequence end, using tickersteps/tinkersteps
            n_tickersteps=5,
        )

        # After the LSTM layer, we add a fully connected output layer
        self.use_cnn = use_cnn
        self.use_prefinal_layer = use_prefinal_layer
        self.pre_fc_out = nn.Linear(n_lstm, n_lstm)
        self.fc_out = nn.Linear(n_lstm, n_outputs)
        self.cnn = nn.Conv1d(kernel_size=kernel_size, in_channels=n_input_features, out_channels=64)

        # self.fc_out.bias.data[:] = -1

    def forward(self, x, true_seq_lens):
        # We only need the output of the LSTM; We get format (samples, n_lstm) since we set return_all_seq_pos=False:
        if self.use_cnn == True:
            x = self.cnn(x)
        lstm_out, *_ = self.lstm1.forward(x,
                                          true_seq_lens=true_seq_lens,  # true sequence lengths of padded sequences
                                          return_all_seq_pos=False  # return predictions for last sequence position
                                          )
        if self.use_prefinal_layer == True:
            lstm_out = self.pre_fc_out(lstm_out)
        net_out = self.fc_out(lstm_out)
        return net_out
