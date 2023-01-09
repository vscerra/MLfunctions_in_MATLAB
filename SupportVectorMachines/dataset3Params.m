function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Initialize variables
Cvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmavals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

error = zeros(length(Cvals),length(sigmavals));
for c = 1:length(Cvals)
    for s = 1:length(sigmavals)
model= svmTrain(X, y, Cvals(c), @(x1, x2) gaussianKernel(x1, x2, sigmavals(s))); 
predictions = svmPredict(model,Xval);
error(c,s) = mean(double(predictions ~= yval));
    end 
end

[a,b] = find(error==(min(min(error))));

C = Cvals(a);
sigma = sigmavals(b);

end
