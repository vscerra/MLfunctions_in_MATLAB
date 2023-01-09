function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% initialize outputs
all_theta = zeros(num_labels, n+1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 

for i = 1:num_labels
    temp_y = y==i;
    initial_theta = zeros(n+1,1);
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50,'Display','off');
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
   all_theta(i,:) = fmincg(@(t)(lrCostFunction(t,X,temp_y,lambda)),...
       initial_theta,options);
end

end
