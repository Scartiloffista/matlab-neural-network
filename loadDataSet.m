function [input, target] = loadDataSet()

input = load('immagini.mat');
input = input.data;
labels = load('labels.mat');
labels = labels.data;
target = 0.*ones(10, size(labels, 1));
for n = 1: size(labels, 1)
    target(labels(n) + 1, n) = 1;
end