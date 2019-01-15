# CNN
## Different architectures of convolutional neural nets tackling Sarcasm Detection
CNN: a single convolution layer CNN
CNN_MLP: a single convolution layer followed by three fully connectedd layers
CNN_3_MLP: three sequential convolution layers followed by three fully connectedd layers
CNN_3_diff_filter_size: three sequential convolution layers followed by a single fully connectedd layer
CNN_and_LSTM: a single convolution layer followed by an LSTM layer


### data_helper.py 
Helper class containing methods for 
  1. loading dataset, 
  2. loding GloVe word embeddings
  3. creating an embedding matrix
### two_K_data.csv
A labelled dataset of 5,344 rows consisting of tweets pulled from twitter API


