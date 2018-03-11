function [precision, recall, right, wrong, dbb] = precisionAndRecall(x, t, net)

[~,~,y] = feedForward(x, net.W, net.B, net.activationF, net.outputF, net.layers);
dbb = [y',t'];
y = y';

tp = 0;
tn = 0;
fp = 0;
right = 0;
wrong = 0;
fn = 0;
for i=1:size(x,2)
    if (y(i) <=0.5)
        y1 = 0;
    else
        y1 = 1;
    end
    if(y1 == t(i) && y1 == 1)
        tp = tp + 1;
        right = right + 1;
    elseif (y1 == t(i) && y1 == 0)
        tn = tn + 1;
        right = right + 1;
    elseif (y1 ~= t(i) && y1 == 1)
        fp = fp +1;
        wrong = wrong + 1;

    else
        fn = fn +1;
        wrong = wrong + 1;
    end
end

recall = tp/(tp+fn);
precision = tp/(tp+fp);