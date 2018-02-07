function dis=pairwise_distance(X,Y)
% compute distance between each line of X and each line of Y
% X and Y does not need to contains same number of lines
dis= sum(X.^2,2)*ones(1,size(Y,1)) + ...
    ones(size(X,1),1)*sum(Y'.^2,1) - 2*X*Y' ;
    
