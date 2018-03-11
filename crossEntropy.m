function [error, gradient] = crossEntropy(y, t )
error = -1 .* sum(t .* log(y) + (1-t) .* log(1-y));
gradient =  ((y-t) ./ (y .* (1-y)));
end