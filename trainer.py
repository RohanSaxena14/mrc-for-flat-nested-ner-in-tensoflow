import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
from bertLayer import BertLayer
from dataGenerator import DataGenerator

max_seq_length = 128
dropout = 0.3
lr_rate = 0.01

#this function will give span matching predictions
def span_matrix_func(tensor):
    global max_seq_length
    embeddings = tensor
    start_expand = K.tile(K.expand_dims(embeddings, 2), [1, 1, max_seq_length, 1])
    end_expand = K.tile(K.expand_dims(embeddings, 1), [1, max_seq_length, 1, 1])

    span_matrix = K.concatenate([start_expand, end_expand], 3)         
    
    return span_matrix


def accuracy(y_true, y_pred):
    mask = y_true[1]
    y_t = y_true[0]
    y_pred = y_pred * mask
    accuracy = K.mean(K.all(y_pred == y_t, axis = 1))
    return accuracy

def recall(y_true, y_pred):
    mask = y_true[1]
    y_true = y_true[0]
    y_pred = y_pred * mask
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    mask = y_true[1]
    y_true = y_true[0]
    y_pred = y_pred * mask
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def bce(y_true, y_pred):
    with open('log', 'w') as fp:
      fp.write(str(y_true[0].shape) + " " + str(y_pred.shape))
    mask = y_true[1]
    y_t = y_true[0]
    loss = K.binary_crossentropy(y_t, y_pred)
    loss = mask*loss
    return loss


def build_function(max_seq_length, dropout):
    input_ids = layers.Input(shape = (max_seq_length), name = "input_ids")
    input_mask = layers.Input(shape = (max_seq_length), name = "input_mask")
    segment_ids = layers.Input(shape = (max_seq_length), name = "segment_ids")
    bert_input = [input_ids, input_mask, segment_ids]
    gs_folder_bert = "bert_en_uncased_L-12_H-768_A-12_3"
    bert_output = BertLayer(path_to_bert=gs_folder_bert)(bert_input)

    #start_predictions_layer = layers.Dense(2, activation='softmax', name = "start_prediction_layer")
    #end_predictions_layer = layers.Dense(2, activation='softmax', name = "end_prediction_layer")

    start_predictions_layer = layers.Dense(1, activation='sigmoid', name = "start_prediction_layer")
    end_predictions_layer = layers.Dense(1, activation='sigmoid', name = "end_prediction_layer")

    start_logits = layers.TimeDistributed(start_predictions_layer, name = "start_logits")(bert_output)
    end_logits = layers.TimeDistributed(end_predictions_layer, name = "end_logits")(bert_output)

    span_matrix = layers.Lambda(span_matrix_func, name = "span_matrix")(bert_output)
    '''span_logits = layers.Conv2D(
                                1,
                                1,
                                input_shape = (max_seq_length, max_seq_length, 2*768),
                                activation="relu",
                                name = "span_logits"
                                )(span_matrix)'''
    span_layer_1 = layers.Dense(768*2, input_shape = (max_seq_length, max_seq_length, 768*2), activation = 'relu', name = "span_dense_1")(span_matrix)
    span_drop_layer_1 = layers.Dropout(dropout, input_shape = (max_seq_length, max_seq_length, 768*2), name = "span_drop_1")(span_layer_1)
    #span_layer_2 = layers.Dense(768*2, input_shape = (max_seq_length, max_seq_length, 768*2), activation = 'relu', name = "span_dense_2")(span_drop_layer_1)
    #span_drop_layer_2 = layers.Dropout(dropout, input_shape = (max_seq_length, max_seq_length, 768*2), name = "span_drop_2")(span_layer_2)
    span_logits = layers.Dense(1, input_shape = (max_seq_length, max_seq_length, 768*2), name = "span_dense_3", activation = 'sigmoid')(span_drop_layer_1)

    flat_span = layers.Flatten(name = "span_flat")(span_logits)
    flat_start = layers.Flatten(name = "start_flat")(start_logits)
    flat_end = layers.Flatten(name = "end_flat")(end_logits)
    
    outputs = [flat_start, flat_end, flat_span]

    model = models.Model(inputs = bert_input, outputs = outputs)

    return model



model = build_function(max_seq_length, dropout)

'''
losses = {
    "span_flat" : bce_span,
    "start_flat" : 'binary_crossentropy',
    "end_flat" : 'binary_crossentropy',
}
'''

losses = {
    "span_flat" : 'binary_crossentropy',
    "start_flat" : 'binary_crossentropy',
    "end_flat" : 'binary_crossentropy',
}

losses_weights = {
    "span_flat" : 0.34,
    "start_flat" : 0.5,
    "end_flat" : 0.5,
}

#optimizer = tf.keras.optimizers.SGD(learning_rate=lr_rate)
optimizer = tf.keras.optimizers.Adam()

model.compile(loss = losses, loss_weights = losses_weights, optimizer= optimizer , metrics = ['accuracy'])
print(model.summary())

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
)