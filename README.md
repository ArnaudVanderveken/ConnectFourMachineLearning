# ConnectFourMachineLearning

This project is an attempt at comparing different machine learning strategies in the context of a game, in this case Connect Four.

## Disclaimer
This project is unfinished, and the AI does not behave correctly. Some of the mentionned features may not be implemented yet.

## How does it work
When starting the program, you will be prompted to choose between these options:
- Play : You'll be able to play Connect4 against the AI in memory.
- Train : Train the AI in memory.
- Save : Save the AI in memory to a *.aidata* file.
- Quit : Exit program.

In the cases "Play" and "Train", you'll be then prompted to choose between these options, regarding th AI to use:
- Create : Create a new AI, thus deleting the one in memory if existing.
- Load : Load an AI from a *.aidata* file, thus deleting the one in memory if existing.
- Keep : Keep the current AI in memory, as long as there is one.

Creating a new AI requires to enter the following parameters:
- Learning method : QLearning or TDLambda.
- Epsilon : Parameter for the Epsilon-greedy exploration strategy.
- Learning Rate : Parameter that controls the magnitude of each change in the weights matrix.
- Lambda : (only if TDLambda has been chosen) Parameter that controls the trace matrices for TDLambda.

In the case of training an AI, you'll be eventually prompted to enter the amount of training games.

## Theory and concepts behind it

### Learning process
In the beginning, the AI will mostly play random moves. After each move, the neural network gets updated based on the delta between the win probability of the state pre-move and post-move. The netwok gets updated one more time at the end of a game with a positive reward in case of a win, and a negative one in case of a loss. These more consequent changes in the network will gradually get reflected on the rest of the network throughout every games.

Since this method will encourage the AI to play the "best" move every time, an exploration parameter is added in order to allow the AI to play a random move from time to time, to discover new game states. This paramter, called *Epsilon*, is a float between 0.0 and 1.0, representing the probability of the AI playing a random move (the higher, the more exploration is allowed). This approach is called *Epsilon-Greedy*

Another parameter of the network is its *Learning Rate*. This float can hold a value between 0.0 and 1.0 and controls how consequent are the changes in the weights of the netwok. Close to 0.0, the changes will be very subtle to none if the learning rate is set to 0.0. On the other hand, approaching 1.0 will result in bigger changes in the weights. Using a too high learning rate can result in network saturation, making it unusable.

### Learning methods
When creating a new AI, you have the option to choose between two learning methods: Qlearning, and Temporal-Difference Lambda (aka TDLambda)

#### QLearning
This method relies on the [QLearning Algorithm](https://en.wikipedia.org/wiki/Q-learning). To update the weight matrices of the network, QLearning will compare the win probability (optained by passing a game state through the neural network) of the state pre-move and the state corresponding to the best possible move, even if the AI plays a random move due to exploration. This gives the advantage not to penalize the networ for exploring other potential solutions.

This delta in the probabilities will be used as reward. To propagate this reward in the network, we use the [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) principle, meaning we start at the output layer of the network and moving backwards through the layers towards the input layer.

#### TDLambda
[TDLambda](https://en.wikipedia.org/wiki/Temporal_difference_learning) is somewhat simial to the QLearning approach, but differs on two points. First it uses the actual move played to determine the probability delta, instead of the best move like in QLearning.

But the biggest difference resides in the use of Trace matrices, one per weight matrix in the network. Reset to 0 before each game, those trace matrices are used as a memory of the prior updates applied to the network. Instead of directly update the weights of the matrices, those changes are added to the trace matrix, then this matrix is used to update the weights. Before adding the new change to the trace matrix, the previous value stored in it gets multiplied by *Lambda*, the factor that controls how long each change will affect future updates in the weights. The closer to 0.0 lambda is, the shorter term is the memory, the closer to 1.0, the longer. Two remarcable values: 0.0 will result in an erasure of the memory in between each move, thus results in a similar update as if directly updating the weights; and 1.0 which preserves in memory all updates that happened throughout the entire game, thus corresponding to the [Monte-Carlo](https://levelup.gitconnected.com/fundamental-of-reinforcement-learning-monte-carlo-algorithm-85428dc77f76) reinforcement method.

### Wait, this is a two player game !
Indeed, but this does not prevent the AI to train against itself using the same neural network and weight matrices! You can consider that the best move for the player two is actually the worst move for the player one, and because our neural network only give the win probability for each given state, it's up to us to decide wether using the highest or the lowest in order to determine the played move.

## Neural Network Architecture
### Game state
A game state is a representation of the Connect Four's grid in the shape of a binary vector. The grid is made out of six rows and seven columns, making fourty-two cells. Each cell can be in three states: empty, occupied by a player-one token, or a player-two token. Because of this, the state vector has a dimension of eighty-four, the first fourty-two elements representing the player-one's token positions, and the remaining elements the player-two's.

### Neural network
#### Input layer
The input layer consists of a one by eighty-four matrix, thus matching the dimensions of a game state vector, allowing them to be bassed as inputs to the network.

#### Hidden layer
This architecture owns a single hidden layer, whose size can be chosen freely. In this case I've chosen it to be made out of eighty-four neurons as well. This also gives us the dimensions of our first weight matrix between the input and hidden layers: an eighty-four by eighty-four matrix.

#### Output layer
The output layer consists of a single neuron. In concequences, the second weight matrix will be an eighty-four by one matrix.

#### Weights initialization
The weights are initialized using a normal distribution with a mean of 0.0 and standard deviation of 0.001. Zero-initialization of these matrix would result in a non-evoluting network, as 0 is absorbant in multiplication. The weights would always remain 0. Using such a distribution allows to dodge that issue, while keeping the weights close to 0, having the least impact on the first moves of the AI.

### Activation function
In order for the neural network to function properly, an activation function is required. In this case, [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function#:~:text=A%20sigmoid%20function%20is%20a,shaped%20curve%20or%20sigmoid%20curve.&text=Sigmoid%20functions%20most%20often%20show,is%20from%20%E2%88%921%20to%201.) has been used on each layer. Other options are available, like [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) or the hyperbolic tangent, each having their pros and cons depending of the use case of the network. In this case, sigmoid has been chosen because its image domain ranges from 0 to 1, perfect to represent a probability. Different activation function can be used on each layer.

## Requirements
The libraries used are:
- Eigen : 'Eigen 3.4.0' is used for the matrices and related operations.
- EigenRand : 'EigenRand 0.4.0-alpha' is an add-on used for random distributions.

## References and Links
- Neural network : https://en.wikipedia.org/wiki/Artificial_neural_network
- QLearning : https://en.wikipedia.org/wiki/Q-learning
- TDLambda : https://en.wikipedia.org/wiki/Temporal_difference_learning
- Monte-Carlo : https://levelup.gitconnected.com/fundamental-of-reinforcement-learning-monte-carlo-algorithm-85428dc77f76
- Backpropagation : https://en.wikipedia.org/wiki/Backpropagation
- Sigmoid : https://en.wikipedia.org/wiki/Sigmoid_function#:~:text=A%20sigmoid%20function%20is%20a,shaped%20curve%20or%20sigmoid%20curve.&text=Sigmoid%20functions%20most%20often%20show,is%20from%20%E2%88%921%20to%201.
- ReLU : https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
