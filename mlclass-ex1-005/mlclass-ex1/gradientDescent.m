function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
% gradientDescent(X,y,theta,0.01,12);

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

tt = zeros(length(theta),1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    for j = 1:length(theta)
        dd = 0;
        for i = 1:m
            h = theta' * X(i,:)';
            dd = dd + (h - y(i)) * X(i,j);
        end;
        tt(j) = theta(j) - alpha * dd / m;
    end;
    theta = tt;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % disp (J_history(iter));

end

% disp(J_history(num_iters));
% plot(1:num_iters,J_history)
end
