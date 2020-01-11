import tensorflow as tf
from tensorflow import keras

from data import retrieve_dataset
from data import CLASSES
from model import model

BATCH_SIZE = 64
INPUT_SHAPE = (50, 200, 3)

def format_target_y(image, labels):
	sets = tuple(label for label in labels)
	return (image,) + sets

def set_shape(image, out_1, out_2, out_3, out_4, out_5):
	image.set_shape((50, 200, 3))
	out_1.set_shape(())
	out_2.set_shape(())
	out_3.set_shape(())
	out_4.set_shape(())
	out_5.set_shape(())
	return image, (out_1, out_2, out_3, out_4, out_5)

def compile_model(model):
	model.compile(
		optimizer='adadelta', loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])


dataset = retrieve_dataset('../images/*.png')
dataset = dataset.map(lambda i, l: tf.py_function(
	func=format_target_y,
	inp=[i, l],
	Tout=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]))
dataset = dataset.map(set_shape).shuffle(buffer_size=1000)
dataset = dataset.batch(BATCH_SIZE)


val_set = retrieve_dataset('../images/*.jpg')
val_set = val_set.map(lambda i, l: tf.py_function(
	func=format_target_y,
	inp=[i, l],
	Tout=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32]))
val_set = val_set.map(set_shape)
val_set = val_set.batch(16)

def play(dataset, val_set):
	m = model(INPUT_SHAPE, len(CLASSES))
	compile_model(m)

	callbacks = [
    		keras.callbacks.ModelCheckpoint(
        	filepath='./checkpoints/mymodel_{epoch}.h5',
        	# Path where to save the model
        	# The two parameters below mean that we will overwrite
        	# the current checkpoint if and only if
	        # the `val_loss` score has improved.
       		save_best_only=True,
		save_weights_only=True,
		monitor='val_loss',
		verbose=1)
	]

	m.fit(dataset, epochs=3, callbacks=callbacks, validation_data=val_set)

if __name__ == '__main__':
	play(dataset, val_set)
