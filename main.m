 clear;
% load data set
[inputValues, targetValues] = loadDataSetBinary();

% hyper-parameters
%eta = 0.2; crossORsquared = 0; strategy = 0;
%eta = 0.001; crossORsquared = 0; strategy = 1;
eta = 0;crossORsquared = 0; strategy = 2;

% (batchSize mod 4) must be 0
sizes=[196 25 1];
for batchSize=[200, 800, 2000, 4000, 8000, 20000, 40000]
% for batchSize=[2000]

    risposta = 'Stiamo usando un set di %d elementi';
    str = sprintf(risposta, batchSize);
    disp(str);
    
    if crossORsquared == 0
        errorFunction = @crossEntropy;
        net = newNetwork(sizes, @sigmoid, @derSigmoid, @sigmoid, @derSigmoid, eta);
        %     str = 'Stiamo usando *Cross entropy* come funzione di errore';
        %     disp(str);
    else
        errorFunction = @squaredError;
        net = newNetwork(sizes, @sigmoid, @derSigmoid, @identity, @derIdentity, eta);
        %     str = 'Stiamo usando *Squared Error* come funzione di errore';
        %     disp(str);
    end
    % split data set
    startTrainingSet = 1;
    endTrainingSet = startTrainingSet + (batchSize / 2) - 1;
    startValidationSet = endTrainingSet + 1;
    endValidationSet = startValidationSet + (batchSize / 4) - 1;
    startTestSet = endValidationSet + 1;
    endTestSet = startTestSet + (batchSize / 4) - 1;
    trainingSet = inputValues(:, startTrainingSet:endTrainingSet);
    validationSet = inputValues(:, startValidationSet:endValidationSet);
    testSet = inputValues(:, startTestSet:endTestSet);
    trainingSetLabels = targetValues(:, startTrainingSet:endTrainingSet);
    validationSetLabels = targetValues(:, startValidationSet:endValidationSet);
    testSetLabels = targetValues(:, startTestSet:endTestSet);
    
    % for rprop
    oldDerEdW = cell(size(sizes,2),1);
    oldDiffW = cell(size(sizes,2),1);
    oldDiffB = cell(size(sizes,2),1);
    oldDerEdB = cell(size(sizes,2),1);
    oldDeltaW = cell(size(sizes,2),1);
    oldDeltaB = cell(size(sizes,2),1);
    for i=2:size(sizes,2)
        oldDerEdW{i} = zeros([sizes(i) sizes(i-1)]);
        oldDerEdB{i} = zeros([sizes(i) 1]);
        oldDeltaW{i} = ones([sizes(i) sizes(i-1)]) ./ 10;
        oldDeltaB{i} = ones([sizes(i) 1]) ./ 10;
        oldDiffW{i} = ones([sizes(i) sizes(i-1)]) ./ 10;
        oldDiffB{i} = ones([sizes(i) 1]) ./ 10;
    end
    
    e=1;
    tic
    while(true)
        c = 1;
        if strategy == 0 % ONLINE
            for i=1:size(trainingSet,2)
                [A,Z,output] = feedForward(trainingSet(:, i), net.W, net.B, net.activationF, net.outputF, net.layers);
                [~, gradient] = errorFunction(output, trainingSetLabels(:, i));
                [derEdB, derEdW] = backPropagation(gradient,A,Z,net.W,net.layers,net.dOutputF, net.dActivationF);
                [net.W, net.B] = gradientDescent(derEdB, derEdW, net.eta, net.B, net.W, net.layers,1);
            end
            [~,Z,output] = feedForward(trainingSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, ~] = errorFunction(output, trainingSetLabels);
            trainingError(e) = sum(error);
            
            [A,Z,output] = feedForward(validationSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, ~] = errorFunction(output, validationSetLabels);
            validationError(e) = sum(error);
            
        elseif strategy == 1 % BATCH
            [A,Z,output] = feedForward(trainingSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, gradient] = errorFunction(output, trainingSetLabels);
            [derEdB, derEdW] = backPropagation(gradient,A,Z,net.W,net.layers,net.dOutputF, net.dActivationF);
            [net.W, net.B] = gradientDescent(derEdB, derEdW, net.eta, net.B, net.W, net.layers, batchSize);
            
            trainingError(e) = sum(error) ;%/ size(trainingSet,2);
            [A,Z,output] = feedForward(validationSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, ~] = errorFunction(output, validationSetLabels);
            validationError(e) = sum(error) ;%/ size(validationSet,2);
            
        else % rprop
            [A,Z,output] = feedForward(trainingSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, gradient] = errorFunction(output, trainingSetLabels);
            [derEdB, derEdW] = backPropagation(gradient,A,Z,net.W,net.layers,net.dOutputF, net.dActivationF);
            [oldDeltaB, net.B, oldDerEdB, oldDiffB] = rprop(net.B, oldDeltaB, oldDerEdB, derEdB, oldDiffB, net.layers);
            [oldDeltaW, net.W, oldDerEdW, oldDiffW] = rprop(net.W, oldDeltaW, oldDerEdW, derEdW, oldDiffW, net.layers);
            
            trainingError(e) = sum(error) ;%/ size(trainingSet,2);
            [A,Z,output] = feedForward(validationSet, net.W, net.B, net.activationF, net.outputF, net.layers);
            [error, ~] = errorFunction(output, validationSetLabels);
            validationError(e) = sum(error) ;%/ size(validationSet,2);
        end
        trainValid = [trainingError', validationError'];
        if(validationError(e) > trainingError(e) && e>2)
            break;
        end
        e = e+1;
    end
    toc
    [precision, recall, ok, notOk, dbb] = precisionAndRecall(testSet,testSetLabels,net);
    
    risposta = 'Precision: %.10f // Recall: %.10f \nRiconosciuti: %d // Non riconosciuti: %d';
    str = sprintf(risposta, precision, recall, ok, notOk);
    disp(str);
    risposta = 'Epoche necessarie per il training = %d';
    str = sprintf(risposta, e);
    disp(str);
    disp('------');
    plot(trainValid);
end
