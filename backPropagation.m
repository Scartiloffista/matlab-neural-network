function [deltas, dW] = backPropagation(gradient, A, Z, W, layers, derivativeO, derivativeH)
deltas{layers} = gradient .* derivativeO(Z{layers});
dW{layers} = deltas{layers} * A{layers-1}';
for i=layers-1:-1:2
   deltas{i} = (W{i+1}' * deltas{i+1}) .* derivativeH(Z{i});
   dW{i} = deltas{i} * A{i-1}';
end

for i=2:layers
     deltas{i} = sum(deltas{i},2);%; ./ size(deltas{layers},2);
end