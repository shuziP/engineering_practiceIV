from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding,SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from sklearn.feature_extraction.text import CountVectorizer


train_df = pd.read_csv(r'input\train_ dataset\nCoV_100k_train.labled.csv',engine ='python')
test_df  = pd.read_csv(r'input\test_dataset\nCov_10k_test.csv',engine ='python')

train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]

train_df['text_cut'] = train_df['微博中文内容'].apply(lambda x:" ".join(jieba.cut(str(x))))
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train_df['text_cut'])
xtrain_count =  count_vect.transform(train_df['text_cut'])
test_df['text_cut'] = test_df['微博中文内容'].apply(lambda x:" ".join(jieba.cut(str(x))))
xtest_count =  count_vect.transform(test_df['text_cut'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_df['text_cut']) + list(test_df['text_cut']))

train_x = tokenizer.texts_to_sequences(train_df['text_cut'])
test_x = tokenizer.texts_to_sequences(test_df['text_cut'])

vocab_size = len(tokenizer.word_index) + 1

maxlen = 30
train_x = pad_sequences(train_x, padding='post', maxlen=maxlen)
test_x = pad_sequences(test_x, padding='post', maxlen=maxlen)

embedding_dim = 50

# model = Sequential()
# model.add(layers.Embedding(input_dim=vocab_size,
#                           output_dim=embedding_dim,
#                           input_length=maxlen))
# model.add(layers.Flatten())

# model.add(layers.Dense(10, activation='relu'))
# model.add(layers.Dense(3, activation='softmax'))
# model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
# model.summary()

##################
model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))

model.add(LSTM(32, activation='relu',
               dropout=0.1,
               ))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
##################
#test_sub = model.predict(test_x)
model.fit(train_x[:300], to_categorical(train_df['情感倾向'].astype(int) + 1)[:300],
         epochs=30,
         batch_size=10)
#test_sub = model.predict(test_x)
from keras.models import load_model

model.save('LSTM_model.h5')

test_sub = model.predict_classes(test_x)
test_sub = test_sub -1
print(test_sub)
a = test_df['微博中文内容']
test_sub_1 = []
for i in test_sub:
    if i == 0:
        test_sub_1.append("--消极情绪")
    elif i == -1:
        test_sub_1.append("--中立情绪")
    elif i == 1:
        test_sub_1.append("--积极情绪")

test_sub_1

for i in range(len(test_sub_1)):
    if i % 5 ==0:
        print(a[i],"----",test_sub_1[i])


c={"微博中文内容":a,
   '情感倾向':test_sub_1}

out_data= pd.DataFrame(c)
# print(out_data)
out_data.to_csv('LSTM_out_put.csv')
# class Dense():
#
#
#     def __init__(self, units,
#                  activation=None,
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)
#         super(Dense, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)
#         self.input_spec = InputSpec(min_ndim=2)
#         self.supports_masking = True
#
#     def build(self, input_shape):
#         assert len(input_shape) >= 2
#         input_dim = input_shape[-1]
#
#         self.kernel = self.add_weight(shape=(input_dim, self.units),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.units,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
#         self.built = True
#
#     def call(self, inputs):
#         output = K.dot(inputs, self.kernel)
#         if self.use_bias:
#             output = K.bias_add(output, self.bias, data_format='channels_last')
#         if self.activation is not None:
#             output = self.activation(output)
#         return output
#
#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) >= 2
#         assert input_shape[-1]
#         output_shape = list(input_shape)
#         output_shape[-1] = self.units
#         return tuple(output_shape)
#
#     def get_config(self):
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer':
#                 regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Dense, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class ActivityRegularization(Layer):
#
#     def __init__(self, l1=0., l2=0., **kwargs):
#         super(ActivityRegularization, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.l1 = l1
#         self.l2 = l2
#         self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)
#
#     def get_config(self):
#         config = {'l1': self.l1,
#                   'l2': self.l2}
#         base_config = super(ActivityRegularization, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape
#
#
# class LSTM(RNN):
#     """
#     """
#
#     @interfaces.legacy_recurrent_support
#     def __init__(self, units,
#                  activation='tanh',
#                  recurrent_activation='sigmoid',
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  recurrent_initializer='orthogonal',
#                  bias_initializer='zeros',
#                  unit_forget_bias=True,
#                  kernel_regularizer=None,
#                  recurrent_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  recurrent_constraint=None,
#                  bias_constraint=None,
#                  dropout=0.,
#                  recurrent_dropout=0.,
#                  implementation=2,
#                  return_sequences=False,
#                  return_state=False,
#                  go_backwards=False,
#                  stateful=False,
#                  unroll=False,
#                  **kwargs):
#         if implementation == 0:
#             warnings.warn('`implementation=0` has been deprecated, '
#                           'and now defaults to `implementation=1`.'
#                           'Please update your layer call.')
#         if K.backend() == 'theano' and (dropout or recurrent_dropout):
#             warnings.warn(
#                 'RNN dropout is no longer supported with the Theano backend '
#                 'due to technical limitations. '
#                 'You can either set `dropout` and `recurrent_dropout` to 0, '
#                 'or use the TensorFlow backend.')
#             dropout = 0.
#             recurrent_dropout = 0.
#
#         cell = LSTMCell(units,
#                         activation=activation,
#                         recurrent_activation=recurrent_activation,
#                         use_bias=use_bias,
#                         kernel_initializer=kernel_initializer,
#                         recurrent_initializer=recurrent_initializer,
#                         unit_forget_bias=unit_forget_bias,
#                         bias_initializer=bias_initializer,
#                         kernel_regularizer=kernel_regularizer,
#                         recurrent_regularizer=recurrent_regularizer,
#                         bias_regularizer=bias_regularizer,
#                         kernel_constraint=kernel_constraint,
#                         recurrent_constraint=recurrent_constraint,
#                         bias_constraint=bias_constraint,
#                         dropout=dropout,
#                         recurrent_dropout=recurrent_dropout,
#                         implementation=implementation)
#         super(LSTM, self).__init__(cell,
#                                    return_sequences=return_sequences,
#                                    return_state=return_state,
#                                    go_backwards=go_backwards,
#                                    stateful=stateful,
#                                    unroll=unroll,
#                                    **kwargs)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#
#     def call(self, inputs, mask=None, training=None, initial_state=None):
#         self.cell._dropout_mask = None
#         self.cell._recurrent_dropout_mask = None
#         return super(LSTM, self).call(inputs,
#                                       mask=mask,
#                                       training=training,
#                                       initial_state=initial_state)
#
#     @property
#     def units(self):
#         return self.cell.units
#
#     @property
#     def activation(self):
#         return self.cell.activation
#
#     @property
#     def recurrent_activation(self):
#         return self.cell.recurrent_activation
#
#     @property
#     def use_bias(self):
#         return self.cell.use_bias
#
#     @property
#     def kernel_initializer(self):
#         return self.cell.kernel_initializer
#
#     @property
#     def recurrent_initializer(self):
#         return self.cell.recurrent_initializer
#
#     @property
#     def bias_initializer(self):
#         return self.cell.bias_initializer
#
#     @property
#     def unit_forget_bias(self):
#         return self.cell.unit_forget_bias
#
#     @property
#     def kernel_regularizer(self):
#         return self.cell.kernel_regularizer
#
#     @property
#     def recurrent_regularizer(self):
#         return self.cell.recurrent_regularizer
#
#     @property
#     def bias_regularizer(self):
#         return self.cell.bias_regularizer
#
#     @property
#     def kernel_constraint(self):
#         return self.cell.kernel_constraint
#
#     @property
#     def recurrent_constraint(self):
#         return self.cell.recurrent_constraint
#
#     @property
#     def bias_constraint(self):
#         return self.cell.bias_constraint
#
#     @property
#     def dropout(self):
#         return self.cell.dropout
#
#     @property
#     def recurrent_dropout(self):
#         return self.cell.recurrent_dropout
#
#     @property
#     def implementation(self):
#         return self.cell.implementation
#
#     def get_config(self):
#         config = {'units': self.units,
#                   'activation': activations.serialize(self.activation),
#                   'recurrent_activation':
#                       activations.serialize(self.recurrent_activation),
#                   'use_bias': self.use_bias,
#                   'kernel_initializer':
#                       initializers.serialize(self.kernel_initializer),
#                   'recurrent_initializer':
#                       initializers.serialize(self.recurrent_initializer),
#                   'bias_initializer': initializers.serialize(self.bias_initializer),
#                   'unit_forget_bias': self.unit_forget_bias,
#                   'kernel_regularizer':
#                       regularizers.serialize(self.kernel_regularizer),
#                   'recurrent_regularizer':
#                       regularizers.serialize(self.recurrent_regularizer),
#                   'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#                   'activity_regularizer':
#                       regularizers.serialize(self.activity_regularizer),
#                   'kernel_constraint': constraints.serialize(self.kernel_constraint),
#                   'recurrent_constraint':
#                       constraints.serialize(self.recurrent_constraint),
#                   'bias_constraint': constraints.serialize(self.bias_constraint),
#                   'dropout': self.dropout,
#                   'recurrent_dropout': self.recurrent_dropout,
#                   'implementation': self.implementation}
#         base_config = super(LSTM, self).get_config()
#         del base_config['cell']
#         return dict(list(base_config.items()) + list(config.items()))
#
#     @classmethod
#     def from_config(cls, config):
#         if 'implementation' in config and config['implementation'] == 0:
#             config['implementation'] = 1
#         return cls(**config)
