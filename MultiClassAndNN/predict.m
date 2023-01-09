function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network with
%3 layers (1 hidden)
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% Initialize outputs 
p = zeros(size(X, 1), 1);

% Go from X to layer 2
X1 = [ones(size(X,1),1) X];
a2 = sigmoid(Theta1*X1')';
%Go from layer 2 to 3
X2 = [ones(size(a2,1),1) a2];
a3 = sigmoid(Theta2*X2');
[~,p] = max(a3,[],1);
p = p';


end
