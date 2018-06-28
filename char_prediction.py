import numpy as np
from pickle import dump
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation
from keras.layers import LSTM
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


sequences = list()
mapping = dict()
# vocab_size=0

train_size = 3000
test_size = 1000

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def load_all_files():
    # load
    lines=[]
    raw_text=[]
    max_len = 0

    for i in range(7):
        in_filename = '/Users/hagardolev/Documents/Computer-Science/Seconed-Year/IML/HACK/IML_Hack/Task2_files/tagged{}.txt'.format(i)
        raw_text += [load_doc(in_filename)]
        lines += raw_text[i].split('\n')

        # integer encode sequences of characters
        chars = sorted(list(set(raw_text[i])))
        for j in chars:
            mapping[j]=ord(j)

        for line in raw_text[i].split('\n'):
            if len(line) == 0:
                continue
            if ord(line[len(line) - 1]) not in [48, 49, 50, 51, 52, 53, 54]:
                continue
            if len(line) > 500:
                continue
            # print(len(line))
            # print(line)
            if max_len < len(line):
                max_len = len(line)

            # integer encode line
            encoded_seq = [ord(char) for char in line]
            # store
            sequences.append(encoded_seq)
            # vocabulary size
    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)
    return max_len , vocab_size


max_len , vocab_size = load_all_files()
sequences = pad_sequences(sequences, maxlen=max_len, truncating='pre')

# exit()

# separate into input and output
# sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
#
# sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
train_indicises = np.random.randint(X.shape[0], size=train_size)
test_indicises = np.random.randint(X.shape[0], size=test_size)
X_train = np.array(X[train_indicises])
y_train = np.array(y[train_indicises])


X_test=np.array(X[test_indicises])
y_test=np.array(y[test_indicises])

# y = to_categorical(y, num_classes=vocab_size)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# [ 0 41 48 49 50 51 52 53 54]
#
data_dim = max_len - 1
timesteps = 1
num_classes = 2


model = Sequential()
model.add(LSTM(50, input_shape=(1, data_dim), return_sequences=True))
# model.add(TimeDistributed(Dense(7, activation='softmax')))
# model.add(TimeDistributed(Activation('softmax')))
#
# model.add(LSTM(30, return_sequences=True,
#                input_shape=(1, data_dim)))  # returns a sequence of vectors of dimension 30
# model.add(LSTM(30, return_sequences=True))  # returns a sequence of vectors of dimension 30
model.add(LSTM(30))  # return a single vector of dimension 30
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy', #loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, batch_size=400, epochs=100, verbose=2)

################# old
# # define model
# model = Sequential()
# model.add(LSTM(75, input_shape=(X.shape[0], X.shape[1])))
# model.add(Dense(vocab_size, activation='softmax'))
# print(model.summary())
# # compile model
# print("hi3")
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit model
# model.fit(X, y, epochs=100, verbose=2)
################# end of old

# # save the model to file
# model.save('model.h5')
# # save the mapping
# dump(mapping, open('mapping.pkl', 'wb'))


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    # for _ in range(n_chars):
        # encode the characters as integers
    encoded = [ord(char) for char in in_text]
    # truncate sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    # one hot encode
    # encoded = to_categorical(encoded, num_classes=len(mapping))
    encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
    # predict character
    yhat = model.predict(encoded, verbose=0)
    # reverse map integer to character
    # out_char = ''
    # for char, index in mapping.items():
    #     if index == yhat:
    #         out_char = char
    #         break
    # append to input
    print(yhat)
    print(chr(yhat[0][0]))
    in_text += chr(yhat[0][0])
    return in_text


# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

score = model.evaluate(X_test, y_test, batch_size=100)
print(score)
# test start of rhyme
print(generate_seq(model, mapping, max_len-1, '      skipBlankLines: true,', 20))
# test mid-line
# print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
# print(generate_seq(model, mapping, 10, 'hello worl', 20))
