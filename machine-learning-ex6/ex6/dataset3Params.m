function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params_set = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
C_temp = params_set;
sigma_temp = params_set;
result_list = zeros(length(params_set) ^ 2, 3);
M = 0;
k = 1;

for i = 1:length(params_set)
    for j = 1:length(params_set)
        model = svmTrain(X, y, C_temp(i), @(x1, x2) gaussianKernel(x1, x2, sigma_temp(j)));
        predictions = svmPredict(model, Xval);
        
        result_list(k, 1) = C_temp(i);
        result_list(k, 2) = sigma_temp(j);
        result_list(k, 3) = mean(double(predictions ~= yval));
        k = k + 1;
    end
end

[M, I] = min(result_list);

C = result_list(I(1,3), 1);
sigma = result_list(I(1,3), 2);





% =========================================================================

end
