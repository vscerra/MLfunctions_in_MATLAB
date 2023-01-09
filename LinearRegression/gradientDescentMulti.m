function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values and outputs
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Perform a single gradient step on the parameter vector theta.
    theta_old = theta;
    delta = zeros(length(theta),1);
    for i = 1:length(theta)
        delta(i) = (1/m)*sum(((X*theta_old)-y).*X(:,i));
    end
    
    theta = theta_old - alpha*delta;
    J_history(iter) = computeCostMulti(X, y, theta);
    
end
end
