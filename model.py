from tensorflow.keras import layers, models


def block(input, filters, kernel, block_count, padding='same', strides=1):
	count = str(block_count)
	X = layers.Conv2D(
		filters, kernel, 
		padding=padding, strides=strides,
		name='conv_' + count)(input)	
	X = layers.BatchNormalization(name='batch_' + count)(X)		
	X = layers.Activation('relu', name='act_relu_' + count)(X)	
	return X

def model(input_shape, num_classes):
	"""Creates the conv net model"""
	input = layers.Input(shape=input_shape, name='input')

	X = block(input, 64, 3, 1)	
	X = layers.MaxPooling2D(data_format='channels_last', name='max_1')(X)
	X = block(X, 64, 3, 2)	
	X = layers.MaxPooling2D(data_format='channels_last', name='max_2')(X)
	X = block(X, 64, 3, 3)	
	X = layers.MaxPooling2D(data_format='channels_last', name='max_3')(X)

	# X = block(X, 128, 3, 4, padding='valid')	
	# X = block(X, 128, 3, 5, padding='valid')	

	X = layers.Flatten(data_format='channels_last')(X)

	outputs = []
	for i in range(5):
		output = layers.Dense(
			num_classes, activation='softmax', name=f'output_{i}')(X)
		outputs.append(output)

	model = models.Model(inputs=input, outputs=outputs)	
	return model


if __name__ == '__main__':
	model = model((50, 200, 3), 36)
	model.summary()
