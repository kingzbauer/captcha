import string

import tensorflow as tf
from matplotlib import pyplot as plt

print("Eager execution: {}".format(tf.executing_eagerly()))

LABELS = list(string.digits + string.ascii_lowercase)
CLASSES = list(range(len(LABELS)))

LABELS_TO_CLASSES = dict(zip(LABELS, CLASSES))
CLASSES_TO_LABELS = dict(zip(CLASSES, LABELS))

def retrieve_dataset(files_path):
	ds = tf.data.Dataset.list_files(files_path)	
	# transform each entry into (image, name)
	def transform(filepath):
		"""retrieves and returns the file contents, the file type e.g
		jpg/png and the label for the specific image.
		The label is the name of the file minus the file extension"""
		filename = tf.strings.split(filepath, '/')[-1]
		label_plus_file_type = tf.strings.split(filename, '.')
		file = tf.io.read_file(filepath)
		return file, label_plus_file_type[1], label_plus_file_type[0]

	def decode_file(file, file_type, label):
		"""Decodes the raw data as either png or jpeg"""
		if file_type == 'jpg':
			return tf.cast(tf.image.decode_jpeg(file), tf.float32)/255.0, label	
		else:
			return tf.cast(tf.image.decode_png(file), tf.float32)/255.0, label

	def transform_string_labels_to_classes(image, string_label):
		return image, tf.cast(
				tf.one_hot(
					[LABELS_TO_CLASSES[label] 
						for label in string_label.numpy().decode()], 
					len(CLASSES)), 
				tf.int32) 
	
	ds = ds.map(transform).map(decode_file).map(
		lambda i, l: tf.py_function(
			func=transform_string_labels_to_classes, 
			inp=[i, l], 
			Tout=[tf.float32, tf.int32]))

	return ds

if __name__ == '__main__':
	ds = retrieve_dataset('../images/*.*')

	for e in ds.take(2):
		print(e[1].numpy())
