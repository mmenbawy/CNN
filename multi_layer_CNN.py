from keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from keras.layers import Conv2D, MaxPooling2D, Embedding
from keras.models import Model
from keras import optimizers
import data_helpers
from datetime import datetime
startTime = datetime.now()
print (startTime)


train_X, val_X, train_y, val_y = data_helpers.load_data_and_labels()
tokenizer, max_length, vocab_size, encoded_train_X, padded_train_X, padded_val_X = data_helpers.give_labels_and_pad_predictors(train_X, val_X, train_y, val_y)
embeddings_index = data_helpers.load_GloVe()
# embedding_matrix = matrix of predefined weights
embedding_matrix = data_helpers.creating_embedding_matrix(tokenizer, vocab_size, embeddings_index, encoded_train_X)

EMBEDDING_DIM = 50

sequence_input = Input(shape=(max_length,), dtype='int32')
print (sequence_input)

embedding_layer = Embedding(vocab_size, 50, input_length=max_length, weights=[embedding_matrix], trainable=False)

#sequence_input = Input(shape=(max_length,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

print (embedded_sequences.shape)
print('--------------')
embedded_sequences = Reshape((max_length, EMBEDDING_DIM, 1))(embedded_sequences)
print (embedded_sequences.shape)

x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
x = MaxPooling2D((max_length - 5 + 1, 1))(x)
# add second conv filter.
y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
y = MaxPooling2D((max_length - 4 + 1, 1))(y)
# add third conv filter.
z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
z = MaxPooling2D((max_length - 3 + 1, 1))(z)
# concate the conv layers
alpha = concatenate([x,y,z])
# flatted the pooled features.
alpha = Flatten()(alpha)
# dropout
#######alpha = Dropout(0.5)(alpha)
# predictions
length_label = len(train_y)+len(val_y)
preds = Dense(1, activation='softmax')(alpha)
# build model
model = Model(sequence_input, preds)
adadelta = optimizers.Adadelta()

model.compile(loss='binary_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
epochs = 2
batch_size = 32
model.fit(padded_train_X, train_y, epochs=5, batch_size=4,verbose=0)
#model.fit(padded_train_X, train_y, epochs=75, batch_size=60, verbose=0)
scores = model.evaluate(padded_val_X, val_y, verbose=0)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(datetime.now()- startTime)
