#Perceptron
#Isadora Ferrao e Sherlon ALmeida

from random import seed
from random import random
from math import exp
from csv import reader
from random import randrange


def initialize_network(n_inputs, n_hidden, n_outputs): #criando uma nova rede neural, aceitando o num de entradas, neuronios e o num de saidas
	network = list()
	hidden_layer= [{'weights':[random() for i in range (n_inputs + 1)]} for i in range (n_hidden)] #para cada camadada oculta sao criados neuronios n_hidden e cada neuronio na camada oculta tem n_inputs + 1 de pesos, um para cada coluna de entrada em um conjunto de dados
	network.append(hidden_layer)
	output_layer =  [{'weights':[random() for i in range (n_hidden + 1)]} for i in range (n_outputs)] #cada neuronio na camada de saida se conecta a
	network.append(output_layer)

	return network		

#seed(1)
#network = initialize_network(2, 1, 2)
#for layer in network:
#	print(layer)

#Propagar para frente
"""
Podemos calcular uma saida de uma rede neural propagando o sinal de entrada
atraves de cada camada ate que a camada de saida produza seus valores
"""

#Ativacao de neuronios
"""
A ativacao de neuronio e calculada como a soma ponderada das entradas
"""

def activate(weights, inputs): #peso e entrada
	activation = weights[-1] #funcao assume q a polatizacao e o ultim peso na lista de pesos
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]

	return activation

#Transferencia de neuronios
"""
Uma vez ativado o neuronio, e transferido a ativacao para ver qual e a saida do neuronio, estamos utilizando a funcao de ativacao sigmoide, tambem chamada de funcao logistica 
Pode pegar qualquer valor de entrada e produzir um numero entre 0 e 1 em uma curva S
"""

def transfer(activation): 
	return 1.0 / (1.0 + exp(-activation)) #equacao sigmoide

#Propagacao direta
"""
A propagacao direta de uma entrada e direta
Todas as saidas de uma camada se tornam entradas para os neuronios da proxima camada
"""

def forward_propagate(network, row): #retorna as saidas da ultima camada, tbm chamada de camada de saida
	inputs = row #linha
	for layer in network: #para cada camada na rede
		new_inputs = [] #cria uma lista de insercoes
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation) #valor de saida do neuronio
			new_inputs.append(neuron['output']) #coletando as saidas de uma camada em uma matriz
		inputs = new_inputs #usando como entrada para a proxima camada
	return inputs

#teste programacao para frente

#network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
#		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
#row = [1, 0, None] #linha
#output = forward_propagate(network, row)
#print(output)

#VOLTAR PROPAGAR ERRO
"""
1- Derivada de transferencia
2-erro de retropropagacao
"""

#Derivada de transferencia 
"""
Dado um valor de saida de um neuronio, e calculado a sua inclinacao
utilizando a funcao de transferencia de sigmoide
"""

def transfer_derivative(output): 
	return output * (1.0 - output)

#Erro de retropropagacao
"""
Calculo do erro para cada neuronio de saida, nos dando o sinal de erro
para se propagar para tras atraves da redes
"""

def backward_propagate_error(network, expected):  #rede e valor de saida esperado para o neuronio
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i+1]:
					error += (neuron['weights'][j] * neuron['delta']) 
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) #determinar o erro do neuronio na camada oculta, onde errors j e o sinal do erro do j-esimo neuronio na camada de saida 

#TESTE - imprimindo a rede apos a retropropagacao do erro
#network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
#expected = [0, 1]
#backward_propagate_error(network, expected)
#for layer in network:
#	print(layer)


#Treinando a rede
"""
A rede e treinada usando gradiente
1- Atualizar pesos
2-rede de trens
"""

#Atualizar pesos
"""
Uma vez que os erros sao calculados eles podem ser usados para atualizar os pesos
"""

def update_weights(network, row, l_rate): #rate e um parametro q deve ser especificado
	for i in range(len(network)):
		inputs = row[:-1]
		if i!=0:
			inputs = [neuron['output'] for neuron in network[i-1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']


#Rede de trens
"""
treinar uma rede por um numero fixo de iteracoes (epocas)
"""

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		

#Prever
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs)) #retorna o indice na saida de rede q tem a maior probabilidade


# TESTE - imprime a saida esperada para cada registro no conjunto de dados de treinamento
#dataset = [[2.7810836,2.550537003,0],
#	[1.465489372,2.362125076,0],
#	[3.396561688,4.400293529,0],
#	[1.38807019,1.850220317,0],
#	[3.06407232,3.005305973,0],
#	[7.627531214,2.759262235,1],
#	[5.332441248,2.088626775,1],
#	[6.922596716,1.77106367,1],
#	[8.675418651,-0.242068655,1],
#	[7.673756466,3.508563011,1]]
#network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
#	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
#for row in dataset:
#	prediction = predict(network, row)
#	print('Expected=%d, Got=%d' % (row[-1], prediction))


# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
	
#Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'seeds.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))




