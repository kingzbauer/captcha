import tensorflow as tf
from matplotlib import pyplot as plt

tf.compat.v1.enable_eager_execution()

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
			return tf.image.decode_jpeg(file), label	
		else:
			return tf.image.decode_png(file), label
	
	ds = ds.map(transform).map(decode_file)

	for i, e in enumerate(ds.take(4)):
		plt.imshow(e[0].numpy())
		plt.show()

if __name__ == '__main__':
	retrieve_dataset('../images/*.*')
