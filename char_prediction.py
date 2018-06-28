import numpy as np
from pickle import dump
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text



sequences = list()
mapping = dict()
vocab_size=0

def load_all_files():
    # load
    lines=[]
    raw_text=[]
    max_len = 0

    for i in range(7):
        in_filename = '/Users/hagardolev/Documents/Computer-Science/Seconed-Year/IML/HACK/IML_Hack/Task2_files/tagged{}.txt'.format(i)
        raw_text += [load_doc(in_filename)]
        lines += raw_text[i].split('\n')
        print(len(lines))
        # print(raw_text[i].split('\n'))


        # integer encode sequences of characters
        chars = sorted(list(set(raw_text[i])))
        for c, j in enumerate(chars):
            mapping[j]=ord(j)

        for line in raw_text[i].split('\n'):
            print(len(line))
            print(line)
            if max_len < len(line):
                max_len = len(line)

            # integer encode line
            encoded_seq = [mapping[char] for char in line]
            # store
            sequences.append(encoded_seq)
            # vocabulary size
            exit()

    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)
    return max_len


max_len = load_all_files()
print(max_len)
# sequences = pad_sequences(sequences, maxlen=max_len)
# print(sequences.shape)

exit()

# separate into input and output
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, epochs=100, verbose=2)

# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))


# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    # generate a fixed number of characters
    for _ in range(n_chars):
        # encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = to_categorical(encoded, num_classes=len(mapping))
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        # predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        # append to input
        in_text += char
    return in_text


# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

# test start of rhyme
print(generate_seq(model, mapping, max_len, '      skipBlankLines: true,', 20))
# test mid-line
# print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
# print(generate_seq(model, mapping, 10, 'hello worl', 20))
