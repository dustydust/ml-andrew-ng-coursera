function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_without_zero = zeros(size(theta) - 1);

% Fill up temp theta without zero parameter
for i = 2:length(theta)
    theta_without_zero(i - 1, 1) = theta(i);
end

J = (-y' * log(sigmoid(X * theta))) - ((1 - y)' * log(1 - sigmoid(X * theta)));
J = J * (1 / m);
J = J + (lambda / (2 * m)) * sum(theta_without_zero .^ 2);

%theta_zero = (1 / m) * ((sigmoid(X * theta) - y)' * X)';
%theta_regularized = (1 / m) * ((sigmoid(X * [0; theta_without_zero]) - y)' * X)' + ((lambda/m) * [0; theta_without_zero]);
%grad = theta_zero + theta_regularized;

h = sigmoid(X * theta);
grad = X' * (h - y);
grad_reg = lambda * theta;
grad_reg(1) = 0;
grad = grad + grad_reg;
grad = grad / m;

% =============================================================

end
