# What is Dimag?
Dimag, Nepali for the brain is an object-oriented neural network framework developed by me in python3. It is the framework that I built during the neural network from scratch course. I used libraries such as NumPy, nnfs, and OpenCV to build this.


## Activation functions 
There are 4 activation functions that can be imported from and be used, they are:
- ReLU Activation function
- Softmax Activation function
- Sigmoid Activation function 
- Linear Activation function
  
## Optimizers
I have built 4 optimizers in the framework so that any can be used as per requirements. They are:
- SGD Optimizer
- AdaGrad Optimizer
- Adam Optimizer
- RMSprop Optimizer

## Loss functions
There are 4 Loss functions in the framework.
- Categorical Crossentropy
- Binary Crossentropy
- Mean Squared Error
- Mean Absolute Error

## Accuracu functions 
- Accuracy Regressional
- Accuracy Categorical
  
## Misc
Moreover, there are functions that are required and some additional functions for convenience.
- Layer_Dense
- Layer_Dropout
- Layer_Input
- Model: for training, saving, and loading the model
- create_mnist_data: for convenience while using mnist data 
  
Other functions such as common loss, load_mnist_data serve as requirements for other main functions.

## Usage

- Clone this repository. <br>
  `git clone https://github.com/B0s0/Dimag.git`<br>
  `cd Dimag`

- Install the required libraries.<br>
  - Unix/macOS:<br>
    `python3 -m pip install -r requirements.txt`<br>
  - Windows:<br>
    `py -m pip install -r requirements.txt`
    
- Add the cloned folder to path at the beginning of your .py file.<br>
  `sys.path.append('<path to cloned repo>')`

- Import necessary objects as per your requirement in your python file. For eg:<br>
  `from dimag import Layer_Dense, Activation_ReLU, Activation_Softmax, np, Activation_Softmax, Model, Loss_CategoricalCrossentropy, Optimizer_Adam, Accuracy_Categorical`

  Alternatively, you can import everything for convenience<br>
  `from dimag import *`

- You can then use the framework to design your model and train it. (Please read `example.md` for more details on this)
  

## Contributing 
If you have any issues please feel free to open an issue.

### For contributing 
- Fork this repository.
- Clone your forked repository to your machine.
- Make necessary changes and commit to your local repository.
- Push those changes to the forked repository.
- Open a pull request requesting those changes in this repository.

If you have any confusions on how to contribute, please check out this awesome article from <a href="https://www.dataschool.io/how-to-contribute-on-github/">Dataschool</a>.
  

## Thanks for using Dimag. Don't forget to star this repository. :smiling_face_with_three_hearts: 	
