function [ A, Z, output] = feedForward(x, W, B, activationF, outputF, layers)
Z = cell(layers,1);
A = cell(layers,1);

A{1} = x;
Z{1} = x;

for i=2:layers-1
    Z{i} = W{i} * A{i-1} + repmat(B{i},1,size(A{i-1},2));
    %Z{i} = W{i} * A{i-1} + B{i};
    
    A{i} = activationF(Z{i});
end
%Z{layers} = W{layers}*A{layers-1} + B{layers};
Z{layers} = W{layers}*A{layers-1} + repmat(B{layers},1,size(A{layers-1},2));

A{layers} = outputF(Z{layers});
output = A{layers};
end

