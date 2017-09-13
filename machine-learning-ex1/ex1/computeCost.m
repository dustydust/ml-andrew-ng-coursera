function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


E = 0;

if (size(theta,1) == 2)
    E = ((X * theta) - y) .^ 2;
elseif (size(theta,1) == 1)
    E = ((X(2) * theta) - y) .^ 2;
end

E = sum(E);

%for i = 1:m
%    example_hypothesis = ((theta(1) + theta(2) * X(i)) - y(i)) ^ 2
%    E = E + example_hypothesis;
%end

J = (1 / (2 * m)) * E;

% =========================================================================

end
