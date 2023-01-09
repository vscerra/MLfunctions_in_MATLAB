function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize variables and outputs
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta.
% Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
g = sigmoid(z);
J = (1/m)*((-y'*log(g))-((1-y')*log(1-g)))+((lambda/(2*m))*sum(theta(2:end,:).^2));
grad(1) = (1/m)*((g-y)'*X(:,1));
for i = 2:length(theta)
    grad(i,1) = ((1/m)*((g-y)'*X(:,i)))+((lambda/m)*theta(i,1));
end

end
