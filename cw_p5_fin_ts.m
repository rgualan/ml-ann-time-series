% Read data from the txt file in a table
T = readtable('ftse_four_year.txt','ReadVariableNames',false,...
    'Delimiter', '\t', 'HeaderLines', 1,...
    'Format','%{MMM dd, yyyy HH:mm}D %f %f %f %f %f %f%%');
%display(T(1:2,:));
TX = flipud(T.Var5); % Extract only the desired column and invert order
N = size(TX,1);
p = 20;

%Use all the data (except the last window) to train the ANN
TX1 = TX(1:N-21); TX2 = TX(N-20:N);
%Ntr = size(TX1,1)-(p+1);
Ntr = size(TX1,1)-(p);
Nts = size(TX2,1)-(p);
%Nts = 1;
Xtr = zeros(Ntr,p); ytr = zeros(Ntr,1); 
Xts = zeros(Nts,p); yts = zeros(Nts,1);
for i=1:Ntr
    Xtr(i,:) = TX1(i:i+p-1)';
    ytr(i) = TX1(i+p);
end
for i=1:Nts
    Xts(i,:) = TX2(i:i+p-1)';
    yts(i) = TX2(i+p);
end
net = feedforwardnet(10);
net = train(net, Xtr', ytr');
yhts = net(Xts'); % validation
figure(1),clf;
plot(TX); hold on; plot(N-Nts:N,yhts,'ro');
