% Time Series Prediction
% The prediction problem is formulated as a regression problem

% generate a time series of 2000 samples.
[ ~, TX ] = mackeyglass_func(2000, 0.2, 0.1, 17, 1.2, 1);
TX= TX(1:2000);
%plot(TX); return;

%Set up training and test data
%Use the firts N = 1500 samples to train a prediction model 
%and the remaining 500 as test data. With p = 20, construct the design 
%matrix and output of a regression problem.
TX1 = TX(1:1500); TX2 = TX(1501:2000);
p = 20;
%Ntr = size(TX1,1)-(p+1);
%Nts = size(TX2,1)-(p+1);
Ntr = size(TX1,1)-p;
Nts = size(TX2,1)-p;
Xtr = zeros(Ntr,p); ytr = zeros(Ntr,1); 
Xts = zeros(Nts,p); yts = zeros(Nts,1);
for i=1:Ntr
    Xtr(i,:) = TX1(i:i+p-1)';
    ytr(i) = TX1(i+p);
end
%plot(ytr);return;
for i=1:Nts
    Xts(i,:) = TX2(i:i+p-1)';
    yts(i) = TX2(i+p);
end

%w = inv(Xtr'*Xtr)*Xtr'*ytr;
w = (Xtr'*Xtr)\(Xtr'*ytr);
yhtr = Xtr*w;
yhts = Xts*w;

figure(1),clf; 
plot(TX, 'LineWidth', 2); hold on; 
%plot(p+1:p+Ntr, yhtr', 'g');
plot(size(TX,1)-size(yhts,1)+1:size(TX,1),yhts,'r', 'LineWidth', 2);
title('TS prediction - Mackey-Glass model', 'FontSize', 16);
legend('Original','Predicted', 'Location','southeast');


%Max Error
max(abs(yhts-yts)),
sqrt(mean((yts - yhts).^2)), %RMSE

pred1 = yhts;
