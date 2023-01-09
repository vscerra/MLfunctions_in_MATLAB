function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Initialize outputs
J = 0;
grad = zeros(size(theta));
dummy_theta = [0;theta(2:end)];

% Compute the cost of a particular choice of theta.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
h = sigmoid(X*theta);
J = ((1/m)*((-y'*log(h))-((1-y')*log(1-h))))+(((lambda/(2*m))*(theta'*dummy_theta)));
grad = (1/m).*(X'*(h-y))+ ((lambda/m)*dummy_theta);

grad = grad(:);

end
