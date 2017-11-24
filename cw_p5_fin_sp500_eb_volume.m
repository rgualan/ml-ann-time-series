% Read data from the txt file in a table
T = readtable('sp500_histdata_5y_1206.csv','ReadVariableNames',true,...
    'Delimiter', ',', 'HeaderLines', 0,...
    'Format','%{yyyy-MM-dd}D %f %f %f %f %f %f');
%display(T(1:2,:));
TC = flipud(T.Close); % Extract only the desired column and invert order (Date order is DESC)
TV = flipud(T.Volume); % Extract only the desired column and invert order (Date order is DESC)
%TC = normalize(TC);
N = size(TC,1);
p = 20;

%Use all the data (except the last window) to train the ANN
%TX1 = TC(1:N-21); 
TX1 = TC(1:N-1); TX2 = TC(N-20:N);
TX3 = TV(1:N-1); TX4 = TV(N-20:N);

%Ntr = size(TX1,1)-(p+1);
Ntr = size(TX1,1)-(p);
Nts = size(TX2,1)-(p);
%Nts = 1;
Xtr = zeros(Ntr,2*p); ytr = zeros(Ntr,1); % twice the columns
Xts = zeros(Nts,2*p); yts = zeros(Nts,1); % twice the columns
for i=1:Ntr
    Xtr(i,:) = [ TX1(i:i+p-1)' TX3(i:i+p-1)' ];
    ytr(i) = TX1(i+p);
end
for i=1:Nts
    Xts(i,:) = [ TX2(i:i+p-1)' TX4(i:i+p-1)' ];
    yts(i) = TX2(i+p);
end

N_ITER = 10;
nets = cell(N_ITER,1);
outputs = zeros(N_ITER,1);
trOutputs = zeros(N_ITER, Ntr);
for i = 1:N_ITER
    net = feedforwardnet(10);
    net = train(net, Xtr', ytr');
    yhtr = net(Xtr')'; %train
    yhts = net(Xts')'; %test
    nets{i} = net;
    outputs(i) = yhts;
    trOutputs(i,:) = net(Xtr');
end


% PLOT OF THE RESULTS
figure(1),clf;
plot(TC, '.-', 'LineWidth', 3); hold on; %original data

% Training predictions with confidence intervals of the training predictions
%plot(size(TX1,1)-size(yhtr,1)+1:size(TX1,1), yhtr, '.-', 'LineWidth', 2); %training prediction
x = p+1:size(TX1,1);
lower = mean(trOutputs) - 1.96*std(trOutputs);
upper = mean(trOutputs) + 1.96*std(trOutputs);
ciplot(lower, upper, x, 'red'); 
plot(x, mean(trOutputs), '.-', 'LineWidth', 2); 

%plot(N-size(yhts)+1:N,yhts,'rx', 'LineWidth', 2);
%plot(N*ones(N_ITER,1),outputs,'rx', 'LineWidth', 2);
errorbar(N,mean(outputs),2*std(outputs), 'x', 'LineWidth', 2);

title('S&P 500 - Close price prediction', 'FontSize', 16);
xlabel('Index', 'FontSize', 14)
ylabel('Close Price', 'FontSize', 14);
legend('Real S&P 500 TS','Tr. Confidence interval','Mean Tr. prediction','Last day prediction', 'Location','northwest');
xlim([1240 1260]); %xlim([1 1260]);
ylim([2120 2240]); %xlim([1 1260]);


%Statistics:
pmean = mean(outputs),
psd = std(outputs),
pmin = min(outputs),
pmax = max(outputs),
%pStError = 2*std(outputs),
RMSE = sqrt(mean(( TC(end)*ones(N_ITER,1) - outputs).^2)), 






