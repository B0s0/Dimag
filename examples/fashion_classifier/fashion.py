# neural network to classify types of clothes using the framework in root directory of this repo

import sys

sys.path.append("../..")

from framework import *

# Label index to label name relation
fashion_mnist_labels = {
    0 : 'T-shirt/top',
    1 : 'Trouser',
    2 : 'Pullover',
    3 : 'Dress',
    4 : 'Coat',
    5 : 'Sandal',
    6 : 'Shirt',
    7 : 'Sneaker',
    8 : 'Bag',
    9 : 'Ankle boot'
}


TRAIN = True #If model is not trained - true, if already trained set to false

# training the model and saving it
if TRAIN == True:

	#preparing data
	X, y, X_test, y_test = create_data_mnist( 'fashion_mnist_images' )
	keys = np.array( range (X.shape[ 0 ]))
	np.random.shuffle(keys)
	X = X[keys]
	y = y[keys]
	X = (X.reshape(X.shape[ 0 ], - 1 ).astype(np.float32) - 127.5 ) / 127.5
	X_test = (X_test.reshape(X_test.shape[ 0 ], - 1 ).astype(np.float32) -
	127.5 ) / 127.5

	#training the model and saving the model
	trainmodel = Model()
	trainmodel.add(Layer_Dense(X.shape[ 1 ], 128 ))
	trainmodel.add(Activation_ReLU())
	trainmodel.add(Layer_Dense( 128 , 128 ))
	trainmodel.add(Activation_ReLU())
	trainmodel.add(Layer_Dense( 128 , 10 ))
	trainmodel.add(Activation_Softmax())

	trainmodel.set(
	  loss=Loss_CategoricalCrossentropy(),
	  optimizer=Optimizer_Adam(decay=1e-4),
	  accuracy=Accuracy_Categorical()
	)

	trainmodel.finalize()

	trainmodel.train(X, y, validation_data = (X_test, y_test),
	epochs = 10 , batch_size = 128 , print_every = 100 )

	trainmodel.save('fashion_mnist.model')


# Reading image and predicting the type fashion loading and using model saved during training

# Read an image
image_data = cv2.imread( 'pants.png', cv2.IMREAD_GRAYSCALE)

# Resize to the same size as Fashion MNIST images
image_data = cv2.resize(image_data, ( 28, 28 ))

# Invert image colors
image_data = 255 - image_data

# Reshape and scale pixel data
image_data = (image_data.reshape( 1, - 1 ).astype(np.float32) -
127.5 ) / 127.5

# Load the model
model = Model.load( 'fashion_mnist.model' )

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[ 0 ]]
print (prediction)