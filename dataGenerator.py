import os
import json
import random
import numpy as np
import tensorflow as tf
from official.nlp.bert import tokenization

class DataGenerator(tf.keras.utils.Sequence):
    '''
    This is a data generator class
    '''
    def __init__(
        self,
        path_to_data = "mrc-ner-mine/data/ONER_data",
        path_to_bert = "mrc-ner-mine/bert_en_uncased_L-12_H-768_A-12_3",
        mode = "train",
        fit = True,
        batch_size = 32,
        max_seq_length = 128,
        shuffle = True,
    ):
        #initializing the tokenizer
        self.tokenizer =  tokenization.FullTokenizer(vocab_file=os.path.join(path_to_bert, "assets/vocab.txt"), do_lower_case=True,)
        
        #This specifies, whether to Train, Test, Validate the model
        if mode in ["train", "test", "dev"]:
            with open(os.path.join(path_to_data, "mrc-ner." + mode)) as fp:
                self.data = json.load(fp)
        else:
            raise ValueError("mode should be one of train, test, dev but given mode " + mode)
        self.path_to_data = path_to_data  #mrc-ner data path
        self.batch_size = batch_size  #batch size
        self.shuffle = shuffle  #Whether to shuffle the data
        self.fit = fit

        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []

        self.start_labels = []
        self.end_labels = []
        self.span_labels = []
    
        self.max_seq_length = max_seq_length
        
        for index, data_point in enumerate(self.data):
            query_tokens = self.tokenizer.tokenize(data_point['query'])
            query_len = len(query_tokens)
            context_tokens = self.tokenizer.tokenize(data_point['context'])
            input_tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + context_tokens

            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            
            input_mask = [1] * len(input_ids)
            input_ids += [0]*(self.max_seq_length - len(input_ids))
            
            input_mask += [0]*(self.max_seq_length - len(input_mask))

            segment_ids = [0] * (query_len + 1) + [1] * (len(context_tokens) + 1)
            segment_ids += [0]*(self.max_seq_length - len(segment_ids))

            start_positions = data_point['start_position']
            end_positions = data_point['end_position']
            span_positions = data_point['span_position']

            #temp_start = [0] * self.max_seq_length * 2
            #temp_end = [0] * self.max_seq_length * 2
            
            temp_start = [0] * self.max_seq_length
            temp_end = [0] * self.max_seq_length
            
            temp_span = np.zeros((self.max_seq_length, self.max_seq_length))
            for index in range(len(start_positions)):
                temp_start[2*(start_positions[index] + query_len + 2) + 1] = 1
                temp_end[2*(end_positions[index] + query_len + 2) + 1] = 1
                temp_start[start_positions[index] + query_len + 2] = 1
                temp_end[end_positions[index] + query_len + 2] = 1

                span_pos = span_positions[index]
                span_start, span_end = list(map(int, span_pos.split(";")))

                temp_span[span_start + query_len + 2][span_end + query_len + 2] = 1
                
            
            self.start_labels.append(temp_start)
            self.end_labels.append(temp_end)
            self.span_labels.append(np.array(temp_span).reshape(self.max_seq_length**2))
            
            self.input_ids.append(input_ids[:self.max_seq_length])
            self.input_mask.append(input_mask[:self.max_seq_length])
            self.segment_ids.append(segment_ids[:self.max_seq_length])


    def __len__(self):
        return int(len(self.data) // self.batch_size)

    def __getitem__(self, index):
        input_ids = np.array(self.input_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        input_mask = np.array(self.input_mask[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        segment_ids = np.array(self.segment_ids[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")

        start_labels = np.array(self.start_labels[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        end_labels = np.array(self.end_labels[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")
        span_labels = np.array(self.span_labels[index*self.batch_size: (index+1)*self.batch_size], dtype="float32")

        span_mask = np.matmul(segment_ids.reshape(-1, self.max_seq_length, 1), segment_ids.reshape((-1, 1, self.max_seq_length)))
        span_mask = span_mask.reshape((-1, self.max_seq_length**2))

        loss_dict = dict(
            span_flat = [span_labels, span_mask][0],
            start_flat = [start_labels, segment_ids][0],
            end_flat = [end_labels, segment_ids][0],
        )
        input_list = [input_ids, input_mask, segment_ids]
        return input_list, loss_dict
    