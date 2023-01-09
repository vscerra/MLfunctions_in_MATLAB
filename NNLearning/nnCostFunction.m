function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.



% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Initialize outputs
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

temp = zeros(size(X,1),1);
for i = 1:m
    %forward propagation for each sample in the training set
    a1 = [1 X(i,:)];
    z2 = a1*Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2*Theta2';
    a3 = sigmoid(z3);
    yi = zeros(num_labels,1);
    yi(y(i))=1;
    delta3 = a3'-yi;
    delta2 = (Theta2'*delta3).*sigmoidGradient([1 z2])';
    Theta2_grad = Theta2_grad + (delta3*a2);
    Theta1_grad = Theta1_grad + (delta2(2:end)*a1);
    temp(i) = (log(a3)*-yi)-(log(1-a3)*(1-yi));
end
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. 
J = ((sum(temp))/m) + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

% Part 3: Implement regularization with the cost function and gradients.
reg_param1 = (lambda/m).*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
reg_param2 = (lambda/m).*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = (Theta1_grad./m)+reg_param1;
Theta2_grad = (Theta2_grad./m)+reg_param2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
