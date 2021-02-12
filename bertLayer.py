import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K

class BertLayer(tf.keras.layers.Layer):
    '''
      This is the bert layer subclass
    '''
    def __init__(
        self,
        path_to_bert = "mrc-ner-mine/bert_en_uncased_L-12_H-768_A-12_3",
        trainable = False,
        pooling = "mean",
        num_fine_tuning_layers = 12,
    ):
        super().__init__()
        self.trainable = trainable #This handles whether you want to train bert layer or not
        self.num_fine_tuning_layers = num_fine_tuning_layers #If you are training , then specify how many layers to fine tune
        self.path_to_bert = path_to_bert #specify the path to bert pretrained model
        self.pooling = pooling #if true, returns only the [CLS] token embedding

    '''
    This function is responsible to initialize the bert model and decide trainable and non trainable parameters
    '''

    def build(self, input_shape):
        self.bert = hub.KerasLayer(self.path_to_bert, trainable=self.trainable, name = "BertLayer") #Loading bert model
        variables = self.bert.variables
        trainable_variables = variables[-self.num_fine_tuning_layers :]     #setting parameters
        non_trainable_weights = variables[: -self.num_fine_tuning_layers]  
        
        for var in trainable_variables:
            self._trainable_weights.append(var)

        for var in non_trainable_weights:
            self._non_trainable_weights.append(var)

    
    def call(self, inputs, training = True):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_word_ids=input_ids,          #This tells you the token ids
            input_mask=input_mask,             #This tells the sentence area apart from the padding area
            input_type_ids=segment_ids,        #This tells the different segment information for multiple sentence input 
        )
        if self.pooling == "first":             #This returs the [CLS] token embeddings
            pooled = self.bert(inputs=bert_inputs)[
                "pooled_output"
            ]
        elif self.pooling == "mean":            #This returns embeddings of all the tokens
            result = self.bert(inputs=bert_inputs)[
                "sequence_output"
            ]

            pooled = result
        
        return pooled