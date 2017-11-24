% Publications tested:
% 1) Wan
% 2) Chaotic time series prediction with residual analysis method
% using hybrid Elman–NARX neural networks

TOT_N = 10000; %Total number of rows of the dataset
%[ ~, TXF ] = mackeyglass_func(600, 0.2, 0.1, 30, 0.9, 6);  % 600/500
%[ ~, TXF ] = mackeyglass_func(1000, 0.2, 0.1, 17, 1.2, 1); % 1000/500
%[ ~, TXF ] = mackeyglass_func(10000, 0.2, 0.1, 17, 1.2, 1); % 10000/5000

% Error grows to fast with this apreach wich is based on [Wan]
%[ ~, TXF ] = mackeyglass_func(TOT_N, 0.2, 0.1, 30, 0.9, 6);  % 5000/25500 

[ ~, TXF ] = mackeyglass_func(TOT_N, 0.2, 0.1, 17, 1.2, 1);  % 5000/2000 >> Good approach
%[ ~, TXF ] = mackeyglass_func(TOT_N);  % 5000/2000 >> Good approach


TXF = TXF(1:TOT_N);
%TXF = normalize(TXF);
p = 20;
figure(1),clf; plot(TXF); hold on; 
%xlim([500, 600]);
%return;

% TRAINING
TX = TXF(1:1500);
Ntr = size(TX,1)-p;
Xtr = zeros(Ntr,p); ytr = zeros(Ntr,1); 
for i=1:Ntr
    Xtr(i,:) = TX(i:i+p-1)';
    ytr(i) = TX(i+p);
end
net = feedforwardnet(20);
% The paper uses 15 
%net = feedforwardnet([10 10]);
%net = feedforwardnet([10 10 10 10 10]);
net = train(net, Xtr', ytr');
yhtr = net(Xtr')'; 



%PREDICTION - FREE RUNNING MODE
%One step at a time
window = yhtr(end-p+1:end);
buffer = window;
FUTURE = window;
%N_ITER = TOT_N - size(TX,1) - p + 1;
N_ITER = TOT_N - size(TX,1);
tic;
for iter=1:N_ITER
    Xts = window';
    step = net(Xts')'; 
    buffer = [buffer; step];
    window = buffer(end-p+1:end);
        
    %stop condition
    if range(window) < 0.0001
        disp('STOP CONDITION FIRED!')
        break;
    end
    
    %reduce buffer size
    buffer = buffer(end-p+1:end);
    FUTURE = [FUTURE; step];
end
toc;
FUTURE=FUTURE(21:end); % Remove first training window
plot(TOT_N-size(FUTURE,1)+1:TOT_N, FUTURE);
title('Mackey-Glass Iterated Prediction', 'FontSize', 16);
xlabel('n', 'FontSize', 14)
ylabel('s(n)', 'FontSize', 14);

error = TXF(end-size(FUTURE,1)+1:end) - FUTURE;
plot(TOT_N-size(FUTURE,1)+1:TOT_N, error);

legend('Mackey-Glass TS','FFNN Iterated Prediction', 'Error','Location','southwest');
%legend('Mackey-Glass Series','FFNN Iterated Prediction','Location','southeast');
%xlim([size(TX,1),size(TXF,1)]);
%xlim([size(TX,1)-100,size(TXF,1)]);
%xlim([size(TXF,1)-500,size(TXF,1)]);
%xlim([size(TX,1)-100,size(TX,1)+100]);
%xlim([size(TXF,1)-200,size(TXF,1)]);

%xlim([1,size(TXF,1)]);
%xlim([1000,2000]);
%xlim([size(TXF,1)-1000,size(TXF,1)]);


%RMSE
sqrt(mean((TXF(end-size(FUTURE,1)+1:end) - FUTURE).^2)), %RMSE




