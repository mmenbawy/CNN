
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import data_helpers

train_X, val_X, train_y, val_y = data_helpers.load_data_and_labels()
tokenizer, max_length, vocab_size, encoded_train_X, padded_train_X, padded_val_X = data_helpers.give_labels_and_pad_predictors(train_X, val_X, train_y, val_y)
embeddings_index = data_helpers.load_GloVe()
# embedding_matrix = matrix of predefined weights
embedding_matrix = data_helpers.creating_embedding_matrix(tokenizer, vocab_size, embeddings_index, encoded_train_X)

		        
 # input_dim, output_dim, input_len        
e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)

model = Sequential()
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_train_X, train_y, epochs=50, batch_size=40,verbose=0)
#model.fit(padded_train_X, train_y, epochs=75, batch_size=60, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_val_X, val_y, verbose=0)
print('Accuracy: %f' % (accuracy*100))
#---7:24 pm 
