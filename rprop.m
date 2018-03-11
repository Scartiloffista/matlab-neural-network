function [delta_c, weights_c, derEW_c, diffW_c] = rprop(oldWeights_c, oldDeltas_c, old_derWdE_c, derEW_c, oldDiffW_c, layers)
etaP = 1.2;
etaM = 0.5;
deltaMAX = 50;
deltaMIN = 0.0001;

diffW_c = cell(layers,1);
delta_c = cell(layers,1);
weights_c = cell(layers,1);

for i=2:layers
    oldWeights = oldWeights_c{i};
    oldDeltas = oldDeltas_c{i};
    old_derWdE = old_derWdE_c{i};
    derEW = derEW_c{i};
    oldDiffW = oldDiffW_c{i};
    
    diffW = zeros(size(oldWeights));
    delta = zeros(size(oldWeights));
    weights =  zeros(size(oldWeights));
    
    % dE/dW (t) * dE/dW (t-1)
    matrix = old_derWdE .* derEW;
    
    %according to sign, get indexes of elements
    %     [r,c] = find(matrix > 0);
    %     ind_pos = [r,c];
    %     [r,c] = find(matrix < 0);
    %     ind_neg = [r,c];
    %     [r,c] = find(matrix == 0);
    %     ind_zer = [r,c];
    ind_pos = find(matrix > 0);
    ind_neg = find(matrix < 0);
    ind_zer = find(matrix == 0);
    
    delta(ind_pos) = min(oldDeltas(ind_pos) .* etaP, deltaMAX);
    delta(ind_neg) = max(oldDeltas(ind_neg) .* etaM, deltaMIN);
    delta(ind_zer) = oldDeltas(ind_zer);
    
    diffW(ind_zer) = -sign(derEW(ind_zer)) .* delta(ind_zer);
    diffW(ind_pos) = -sign(derEW(ind_pos)) .* delta(ind_pos);
    diffW(ind_neg) = oldDiffW(ind_neg);
    
    weights(ind_pos) = oldWeights(ind_pos) + diffW(ind_pos);
    weights(ind_neg) = oldWeights(ind_neg) - oldDiffW(ind_neg);
    weights(ind_zer) = oldWeights(ind_zer) + diffW(ind_zer);
    
    derEW(ind_neg) = 0;
    
    diffW_c{i} = diffW;
    delta_c{i} = delta;
    weights_c{i} = weights;
    oldWeights_c{i} = oldWeights;
    oldDeltas_c{i} = oldDeltas;
    old_derWdE_c{i} = old_derWdE;
    derEW_c{i} = derEW;
    oldDiffW_c{i} = oldDiffW;
end
end