function [ error, gradient ] = squaredError(output, target)
error = 0.5 * sum( ( (output - target).^2 ) ); 
gradient =  (output - target);
end

