import numpy as np
import math

# receptron activation function
def predict(inputs, weights):
    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i*w
    return 1.0 if activation >= 0.0 else -1.0

# each matrix row: up to last row = inputs, last row = y (classification)
def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)  # get predicted classification
        preds.append(pred)
        if pred == matrix[i][-1]:
            num_correct += 1.0
    return num_correct/float(len(matrix))

# each matrix row: up to last row = inputs, last row = y (classification)
def train_weights(matrix, weights, nb_epoch=10, l_rate=1.00, do_plot=False, stop_early=True, verbose=True):

    for epoch in range(nb_epoch):
        cur_acc = accuracy(matrix, weights)

        if cur_acc == 1.0 and stop_early:
            break

        for i in range(len(matrix)):

            # get predicted classificaion
            prediction = predict(matrix[i][:-1], weights)
            # get error from real classification
            error = matrix[i][-1]-prediction
            for j in range(len(weights)): 				 # calculate new weight for each node
                weights[j] = weights[j]+(l_rate*error*matrix[i][j])
    print("\nWeights: ", weights)

    return weights


def getMatrix():
    R = 3
    C = int(input("Please enter how many sets of triplets you would like to test:\n"))
    print("Please enter the number of input samples separated by a space:")
    entries = list(map(int, input().split()))
    matrix = np.array(entries).reshape(R, C)
    col_count = len(matrix[:][0])
    row_count = len(matrix[:])
    total = row_count * col_count
    return matrix


def getMatrixCount():
    R = 3
    C = int(input("Please enter how many sets of triplets you would like to test:\n"))
    print("Please enter the number of input samples separated by a space:")
    entries = list(map(int, input().split()))
    matrix = np.array(entries).reshape(R, C)
    col_count = len(matrix[:][0])
    row_count = len(matrix[:])
    total = row_count * col_count
    return total

# class LogisticRegress:
# 	def __init__(self, no_of_features=2):
# 		"""Initialize parameters to zeros"""
# 		# no of parameters will be equal to no of features
# 		self.parameters = np.zeros((no_of_features), dtype='float')
# 		self.no_of_features = no_of_features

# 	def predict(self, input_data):
# 		"""Predict output given input"""
# 		# return the output
# 		return 1 / ( 1 + math.e ** -np.dot(self.parameters.transpose(), input_data) )


# 	#logistic regression function
# 	def sigmoid(self, t):
# 		return 1 / (1 + math.exp(-t))
# 	def logits(self):
# 		batch_size = 1
# 		alpha = 0.1
# 		iterations = 1000
# 		numbers = getMatrixCount()
# 		while iterations > 0:
# 			for i in range(0,numbers,batch_size):
# 				end = numbers if i + batch_size > numbers else i + batch_size
# 				predictions = np.array( [self.predict(j) for j in getMatrix[i:end]])
# 				for index in range(len(self.parameters)):
# 					derivative = 0
# 					for k in range(batch_size):
# 						if i + k >=  numbers:
# 							break
# 						else:
# 							derivative += ()


# logistic regression function
def sigmoid(t):
    return 1 / (1 + math.exp(-t))


def logistic():
    numbers = getMatrixCount()
    outputs = 0
    trainingCount = numbers * 100
    train_a = 1
    alpha = 0.1
    weights = [0, 0]
    for i in range(numbers):
        weights.append(0)
    currentWeight = 0
    while(train_a <= trainingCount):
        #derivative = (g(x))(1-g(x))
        delta = sigmoid(weights[outputs]) * (1-sigmoid(weights[outputs]))
        weights[outputs] = weights[outputs] + (alpha * delta)
        train_a += 1
        if (outputs >= numbers):
            outputs = 0
        else:
            outputs += 1
    for i in weights:
        i -= 1
        if i > 0:
            print(i)


def perceptron():

    nb_epoch = 10
    l_rate = 1.0

    getMatrix
    weights = [0, 0]  # initial weights specified in problem
    train_weights(getMatrix(), weights=weights,
                  nb_epoch=nb_epoch, l_rate=l_rate)


if __name__ == '__main__':
    user_input = input("Please enter either 'p' or 'l':\n")
    if user_input == "p":
        perceptron()
    elif user_input == "l":
        logistic()
    else:
        print("enter either p or q")
