function [W, B] = gradientDescent(dB, dW, eta, B, W, layers, batchSize)

for l=layers:-1:2
    B{l} = B{l} - (eta) * dB{l};
    W{l} = W{l} - (eta) * dW{l};
end