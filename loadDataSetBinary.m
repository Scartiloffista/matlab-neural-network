function [input, target] = loadDataSetBinary()

input = load('immagini.mat');
input = input.data;
labels = load('labels.mat');
labels = labels.data;
target = 0.*ones(1, size(labels, 1));
for n = 1: size(labels, 1)
    if(labels(n) == 7)
        target(n) = 1;
    else
        target(n) =0;
    end
end