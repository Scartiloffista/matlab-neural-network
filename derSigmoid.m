function [ y ] = derSigmoid( x )
    y = sigmoid(x).*(1 - sigmoid(x));
end