import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        
        #print("Printing x (input)")
        #print(x)
        #print("Printing w (self.weights)")
        #print(self.w)
        return nn.DotProduct(x, self.w)
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dotProductScalar = nn.as_scalar(self.run(x))
        return -1.0 if dotProductScalar < 0 else 1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        converged = False
        epoch_number = 1

        while not converged:
            sampleNumber = 1
            totalIncorrect = 0
            for x, y in dataset.iterate_once(batch_size):
                prediction = self.get_prediction(x)
                correct = nn.as_scalar(y)
                
                if prediction == -1.0 and correct == 1.0:
                    totalIncorrect += 1
                    self.w.update(x, 1.0)
                elif prediction == 1.0 and correct == -1.0:
                    totalIncorrect += 1
                    self.w.update(x, -1.0)
                          
                sampleNumber += 1
            
            print("Completed epoch number " + str(epoch_number) + ", total incorrect: " + str(totalIncorrect))
            epoch_number += 1
            
            if totalIncorrect == 0:
                converged = True
            
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        
        self.batch_size = 1
        self.hidden_layer_width = 7
        
        # Layer 1: This is the W(1) matrix
        self.w1 = nn.Parameter(1, self.hidden_layer_width)
        self.b1 = nn.Parameter(1, self.hidden_layer_width)
        
        self.z1 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        self.z1b = nn.Parameter(self.batch_size, 1)
        
        self.a1 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        
        
        # Layer 2: This is the W(2) matrix
        self.w2 = nn.Parameter(self.hidden_layer_width, self.hidden_layer_width)
        self.b2 = nn.Parameter(1, self.hidden_layer_width)
        
        self.z2 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        self.z2b = nn.Parameter(self.batch_size, 1)
        
        self.a2 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        
        # Layer 3: This is the W(3) matrix
        self.w3 = nn.Parameter(self.hidden_layer_width, self.hidden_layer_width)
        self.b3 = nn.Parameter(1, self.hidden_layer_width)
        
        self.z3 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        self.z3b = nn.Parameter(self.batch_size, 1)
        
        self.a3 = nn.Parameter(self.batch_size, self.hidden_layer_width)
        
        # This is the W(n) matrix
        self.wn = nn.Parameter(self.hidden_layer_width, 1)
        self.bn = nn.Parameter(1, 1)
        
        self.zn = nn.Parameter(self.batch_size, 1)
        self.znb = nn.Parameter(self.batch_size, 1)
        #self.yHat = nn.Parameter(self.batch_size, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        self.z1 = nn.Linear(x, self.w1)
        self.z1b = nn.AddBias(self.z1, self.b1)
        self.a1 = nn.ReLU(self.z1b)
        
        self.z2 = nn.Linear(self.a1, self.w2)
        self.z2b = nn.AddBias(self.z2, self.b2)
        self.a2 = nn.ReLU(self.z2b)
        
        self.z3 = nn.Linear(self.a2, self.w3)
        self.z3b = nn.AddBias(self.z3, self.b3)
        self.a3 = nn.ReLU(self.z3b)
        
        self.zn = nn.Linear(self.a3, self.wn)
        self.znb = nn.AddBias(self.zn, self.bn)
        #self.yHat = nn.ReLU(self.z3b)
        
        return self.znb
        #wx1 = nn.Linear(x, self.w1)
        #wxb1 = nn.AddBias(wx1, self.b1)
        #h1 = nn.ReLU(wxb1)
        
        #wx2 = nn.Linear(h1, self.w2)
        #wxb2 = nn.AddBias(wx2, self.b2)
        #predict2 = nn.ReLU(wxb2)
        #return wxb2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_predict = self.run(x)
        return nn.SquareLoss(y_predict, y)
        

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        alpha = 0.025
        minLoss = 1000
        epoch = 1
        converged = False
        
        
        while not converged:
            counter = 0
            totalLoss = 0.0
            
            for x, y in dataset.iterate_once(self.batch_size):
                print("======================================================")
                print("counter = " + str(counter))
                
                print("x is below: ")
                x.print_node()
                
                print("y is below: ")
                y.print_node()
                
                currentLoss = self.get_loss(x, y)
                
                totalLoss += nn.as_scalar(currentLoss)
                
                ## Layer 1
                
                print("w1:")
                self.w1.print_node()
                
                print("b1:")
                self.b1.print_node()
                
                print("z1:")
                self.z1.print_node()
                
                print("z1b:")
                self.z1b.print_node()
                
                print("a1:")
                self.a1.print_node()
            
                ## Layer 2
            
                print("w2:")
                self.w2.print_node()
                
                print("b2:")
                self.b2.print_node()
                
                print("z2:")
                self.z2.print_node()
                
                print("z2b:")
                self.z2b.print_node()
                
                print("a2:")
                self.a2.print_node()
                
                ### Last Layer
                
                print("wn:")
                self.wn.print_node()
                
                print("bn:")
                self.bn.print_node()
                
                print("zn:")
                self.zn.print_node()
                
                print("znb:")
                self.znb.print_node()
                
                #print("yHat:")
                #self.yHat.print_node()
    
                
                print("Current Loss:")
                print(nn.as_scalar(currentLoss))
                
                step_gradients = nn.gradients(currentLoss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.wn, self.bn])
                
                print("Step gradient w1:")
                print(step_gradients[0].print_node())
                
                print("Step gradient b1:")
                print(step_gradients[1].print_node())
                
                print("Step gradient w2:")
                print(step_gradients[2].print_node())
                
                print("Step gradient b2:")
                print(step_gradients[3].print_node())
                
                print("Step gradient wn:")
                print(step_gradients[4].print_node())
                
                print("Step gradient bn:")
                print(step_gradients[5].print_node())
                
                self.w1.update(step_gradients[0], -alpha)
                self.b1.update(step_gradients[1], -alpha)
                self.w2.update(step_gradients[2], -alpha)
                self.b2.update(step_gradients[3], -alpha)
                self.w3.update(step_gradients[4], -alpha)
                self.b3.update(step_gradients[5], -alpha)
                self.wn.update(step_gradients[6], -alpha)
                self.bn.update(step_gradients[7], -alpha)
                
                print("New Loss:")
                newLoss = self.get_loss(x, y)
                print(nn.as_scalar(newLoss))
                
                print("Alpha:")
                print(alpha)
                
                print("Epoch:")
                print(epoch)
                
                counter += 1
            
            averageLoss = totalLoss / counter
            
            print("Average Loss: " + str(averageLoss))
            
            if averageLoss < 0.02:
                converged = True
            
            alpha *= 0.99
            epoch += 1

class Layer(object):
    """
    Represents a single layer in a neural network.
    """
    
    def __init__(self, input_size, node_count, output_size):
        #object.__init__(self, *args, **kwargs)
        self.x = 3
            

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
