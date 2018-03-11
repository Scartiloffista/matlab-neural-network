function net=newNetwork(sizes, hnFunction, derHnFunction,outFunction, derOutFunction, eta)
% generate random values for weights and biases
for i=2:length(sizes)
    net.W{i} = 1-2*rand(sizes(i),sizes(i-1));
    net.B{i} = 1-2*rand(sizes(i),1);
    
    % for rprop
    net.DeltaW{i} = repmat(0.1, sizes(i), sizes(i-1));
    net.DeltaB{i} = repmat(0.1, sizes(i), 1);
end
net.activationF = hnFunction;
net.eta = eta;
net.dActivationF = derHnFunction;
net.outputF = outFunction;
net.dOutputF = derOutFunction;
net.layers = length(sizes);
return