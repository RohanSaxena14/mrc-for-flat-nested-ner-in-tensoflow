#Loading the saved model
import tensorflow as tf
from dataGenerator import DataGenerator

path_of_model = ""
path_to_bert = "/bert_en_uncased_L-12_H-768_A-12_3"
path_to_data = "**DATA PATH**"

model = tf.keras.models.load_model(path_of_model)


#Making a test data generator
test_data = DataGenerator(mode="train", path_to_bert=path_to_bert, path_to_data=path_to_data)

#Evaluating the accuracy on test data
k = model.evaluate(test_data)