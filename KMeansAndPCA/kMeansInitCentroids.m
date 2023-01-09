function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% Set centroids to randomly chosen examples from
%               the dataset X
%randomly reorder samples:
randidx = randperm(size(X,1)); % this ensures that you don't get two same values
%select the first K samples from the permuted data
centroids = X(randidx(1:K),:);

end

