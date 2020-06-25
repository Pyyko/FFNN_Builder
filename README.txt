I made this project to recap myself of how to make really simple neural networks, it can be improve by
many ways but for now i don't feel motivated.


# What's the project ?

It's a sort of neural network builder. 


It's very limited but here's what you can do :

	# Build.py

		- fully connected neural network
		
			2 choice of activations functions (reLU/sigmoid)
			choose a regularization strenght
			choose a learning rate

		- use the MNIST dataset to train your neural network (you theoretically can use other dataset, see /MNIST_data/README.txt)
			
			you can modify the dataset by using feature scaling or mean normalization on it		
			you can mini-batch your dataset too
			you can see how well is doing your neural network while training with realtime graph (accuracy / cost function)


	# MNIST_the_game.py

	- play with your neural networks in a really simple game, guessing digit against them, maybe one of your neural network will
	be more accurate than you



# What folder is used for what ?

saved_ai : you can save your neural network in there, (MNIST_the_game.py only search for neural network in this file)
stuff : some modules
MNIST_data : MNIST data, in the format : pickle object (you theoretically can use put other dataset in, see /MNIST_data/README.txt)


And... that's it :)

