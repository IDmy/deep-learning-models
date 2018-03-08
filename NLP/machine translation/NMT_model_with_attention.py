from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from utils import *
import matplotlib.pyplot as plt


#============================== Dataset =======================================

#dataset - a list of tuples of (human readable date, machine readable date)
#human_vocab - a python dictionary mapping all characters used in the human readable dates to an integer-valued index 
#machine_vocab - a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with `human_vocab`. 
#inv_machine_vocab - the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
dataset[:10]

#X - a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to $T_x$ values with a special character (< pad >). `X.shape = (m, Tx)`
#Y - a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
#Xoh - one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
#Yoh - one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 
Tx = 20
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])

#=============== Neural machine translation with attention ====================

#Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    #Using repeator to repeat s_prev to be of shape (m, Tx, n_s) so that we can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev)
    #Using concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])
    #Using densor to propagate concat through a small fully-connected neural network to compute the "energies" variable e
    e = densor(concat)
    #Using activator and e to compute the attention weights "alphas"
    alphas = activator(e)
    #Using dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas,a])
    
    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    # Defining the inputs of your model with a shape (Tx,), s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initializing empty list of outputs
    outputs = []
       
    #Defining pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(n_a, return_sequences=True),input_shape=(m, Tx, n_a))(X)
    
    #Iterating for Ty steps
    for t in range(Ty):
    
        #Performing one step of the attention mechanism to get back the context vector at step t 
        context = one_step_attention(a, s)
        #Applying the post-attention LSTM cell to the "context" vector.
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
        
        #Applying Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)        
        outputs.append(out)
        
    #Creating model instance taking three inputs and returning the list of outputs
    model = Model([X,s0,c0], outputs)
    
    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

#============================ Optimiztion =====================================
opt =  Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01) 
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


#============================== Training ======================================

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
out_path = 'output/'
your_date = "Tuesday Apr 27 1983"

for i in range(20):
    model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
    #Visualizing attention mechanism (alphas in one_step_attention)
    attention_map = plot_attention_map(out_path, i, model, human_vocab, inv_machine_vocab, your_date , num = 6, n_s = 64)

#============================== Example =======================================
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    source = np.array(string_to_int(example, Tx, human_vocab)).reshape((1, 30))
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))#.swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("input :", example)
    print("output:", ''.join(output))



