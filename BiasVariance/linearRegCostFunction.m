function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% Initialize outputs 
J = 0;
grad = zeros(size(theta));

% Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.

J = ((1/(2*m))*sum(((X*theta)-y).^2)) + ((lambda/(2*m))*sum(theta(2:end).^2));

grad(1) = (1/m)*sum(((X*theta)-y));
for j = 2:length(theta)
    grad(j) = ((1/m)*sum(((X*theta)-y).*X(:,j))) + ((lambda/m)*theta(j));
end

grad = grad(:);

end
