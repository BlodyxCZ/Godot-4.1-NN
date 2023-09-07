class_name NeuralNetwork
extends Node


var input_layer_size: int
var hidden_layer_size: int
var output_layer_size: int

var weights_input_hidden: Array
var weights_hidden_output: Array
var bias_hidden: Array
var bias_output: Array

var learning_rate: float = 0.1

var best_loss: float = INF
var best_weights_input_hidden: Array
var best_weights_hidden_output: Array
var best_bias_hidden: Array
var best_bias_output: Array

func _init(input_size, hidden_size, output_size):
	input_layer_size = input_size
	hidden_layer_size = hidden_size
	output_layer_size = output_size
	
	weights_input_hidden = []	
	weights_hidden_output = []
	bias_hidden = []
	bias_output = []
	
	randomize_weights_bias()

func randomize_weights_bias():
	# Randomly initialize weights and biases between -1 and 1
	weights_input_hidden.clear()
	weights_hidden_output.clear()
	bias_hidden.clear()
	bias_output.clear()
	
	for i in range(hidden_layer_size):
		var hidden_weights = []
		for j in range(input_layer_size):
			hidden_weights.append(randf_range(-1.0, 1.0))
		weights_input_hidden.append(hidden_weights)
		bias_hidden.append(randf_range(-1.0, 1.0))
	
	for i in range(output_layer_size):
		var output_weights = []
		for j in range(hidden_layer_size):
			output_weights.append(randf_range(-1.0, 1.0))
		weights_hidden_output.append(output_weights)
		bias_output.append(randf_range(-1.0, 1.0))

func sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

func forward_propagation(inputs):
	# Calculate output using forward propagation
	var hidden_layer = []
	var output_layer = []
	
	# Calculate hidden layer values
	for i in range(hidden_layer_size):
		var hidden_sum = 0
		for j in range(input_layer_size):
			hidden_sum += inputs[j] * weights_input_hidden[i][j]
		hidden_sum += bias_hidden[i]
		hidden_layer.append(sigmoid(hidden_sum))
	
	# Calculate output layer values
	for i in range(output_layer_size):
		var output_sum = 0
		for j in range(hidden_layer_size):
			output_sum += hidden_layer[j] * weights_hidden_output[i][j]
		output_sum += bias_output[i]
		output_layer.append(sigmoid(output_sum))
	
	return output_layer

func backward_propagation(inputs, targets):
	# Perform backward propagation and update weights and biases
	var hidden_layer = []
	var output_layer = []
	var hidden_errors = []
	var output_errors = []
	
	# Calculate hidden layer values
	for i in range(hidden_layer_size):
		var hidden_sum = 0
		for j in range(input_layer_size):
			hidden_sum += inputs[j] * weights_input_hidden[i][j]
		hidden_sum += bias_hidden[i]
		hidden_layer.append(sigmoid(hidden_sum))
	
	# Calculate output layer values
	for i in range(output_layer_size):
		var output_sum = 0
		for j in range(hidden_layer_size):
			output_sum += hidden_layer[j] * weights_hidden_output[i][j]
		output_sum += bias_output[i]
		output_layer.append(sigmoid(output_sum))
	
	# Calculate output errors
	for i in range(output_layer_size):
		output_errors.append(targets[i] - output_layer[i])
	
	# Calculate hidden errors
	for i in range(hidden_layer_size):
		var error = 0
		for j in range(output_layer_size):
			error += output_errors[j] * weights_hidden_output[j][i]
		hidden_errors.append(error)
	
	# Update weights and biases
	for i in range(output_layer_size):
		for j in range(hidden_layer_size):
			weights_hidden_output[i][j] += learning_rate * output_errors[i] * output_layer[i] * (1 - output_layer[i]) * hidden_layer[j]
	
	for i in range(hidden_layer_size):
		for j in range(input_layer_size):
			weights_input_hidden[i][j] += learning_rate * hidden_errors[i] * hidden_layer[i] * (1 - hidden_layer[i]) * inputs[j]
	
	for i in range(output_layer_size):
		bias_output[i] += learning_rate * output_errors[i] * output_layer[i] * (1 - output_layer[i])
	
	for i in range(hidden_layer_size):
		bias_hidden[i] += learning_rate * hidden_errors[i] * hidden_layer[i] * (1 - hidden_layer[i])

func train(inputs, targets, epochs, validation_data):
	# Train the neural network using backpropagation
	for epoch in range(epochs):
		for i in range(inputs.size()):
			var input_data = inputs[i]
			var target_data = targets[i]
			backward_propagation(input_data, target_data)
	# Save the best model if validation loss is improved
	var validation_loss = calculate_loss(validation_data)
	if validation_loss < best_loss:
		best_loss = validation_loss
		best_weights_input_hidden = weights_input_hidden.duplicate()
		best_weights_hidden_output = weights_hidden_output.duplicate()
		best_bias_hidden = bias_hidden.duplicate()
		best_bias_output = bias_output.duplicate()

func calculate_loss(data):
	# Helper function to calculate the loss (e.g., mean squared error)
	# using the current weights and biases
	var total_loss = 0
	for i in range(data.size()):
		var input_data = data[i][0]
		var target_data = data[i][1]
		var prediction = forward_propagation(input_data)
		var loss = calculate_mean_squared_error(prediction, target_data)
		total_loss += loss
	return total_loss / data.size()

func calculate_mean_squared_error(prediction, target):
	var error = 0
	for i in range(prediction.size()):
		error += (prediction[i] - target[i]) ** 2
	return error / prediction.size()

func predict(inputs):
	# Make predictions using the trained neural network
	return forward_propagation(inputs)

func save_best_model(filename: String):
	# Save the best model weights and biases to a file
	var model_data = {
		"input_hidden": best_weights_input_hidden,
		"hidden_output": best_weights_hidden_output,
		"bias_hidden": best_bias_hidden,
		"bias_output": best_bias_output
	}
	var file = FileAccess.open(filename, FileAccess.WRITE)
	var error = FileAccess.get_open_error()
	if error == OK:
		file.store_var(model_data)
		file.close()
		print("Best model saved to: ", filename)
	else:
		print("There was an error saving file to: ", filename)
