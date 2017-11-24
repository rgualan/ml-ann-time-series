% Read data from the txt file in a table
T = readtable('sp500_histdata_5y_1206.csv','ReadVariableNames',true,...
    'Delimiter', ',', 'HeaderLines', 0,...
    'Format','%{yyyy-MM-dd}D %f %f %f %f %f %f');
%display(T(1:2,:));
TC = flipud(T.Close); % Extract only the desired column and invert order (Date order is DESC)
TV = flipud(T.Volume); % Extract only the desired column and invert order (Date order is DESC)
%TC = normalize(TC);
%TV = normalize(TV);
N = size(TC,1);
p = 20;


%Use all the data (except the last window) to train the ANN
TX1 = TC(1:N-21); TX2 = TC(N-20:N);
TX3 = TV(1:N-21); TX4 = TV(N-20:N);
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
net = feedforwardnet(20);
net = train(net, Xtr', ytr');
yhts = net(Xts'); % validation
figure(1),clf;
plot(TC, '+-', 'LineWidth', 2); hold on; plot(N-size(yhts)+1:N,yhts,'rx', 'LineWidth', 2);
title('S&P 500 - Close price prediction', 'FontSize', 16);
xlabel('Index', 'FontSize', 14)
ylabel('Close Price', 'FontSize', 14);
legend('Original','Predicted', 'Location','southeast');
xlim([1220 1260]);


ERROR=abs(yhts-TC(end));
display(ERROR);