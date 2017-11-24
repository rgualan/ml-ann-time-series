TOT_N = 600; %Total number of rows of the dataset
[ ~, TXF ] = mackeyglass_func(TOT_N, 0.2, 0.1, 30, 0.9, 6);
TXF = TXF(1:TOT_N);
p = 20;
%figure(1),clf; plot(TXF);return;
title('Mackey-Glass(30) Time Series', 'FontSize', 16);
xlabel('k', 'FontSize', 14)
ylabel('x', 'FontSize', 14);

% TRAINING
TX = TXF(1:500);
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
%Ploting
plot(21:size(TX,1),yhtr, 'LineWidth', 2);
% max(abs(yhts-yts)), %Max Error
%return;



%PREDICTION - FREE RUNNING MODE
%One step at a time
window = yhtr(end-p+1:end);
buffer = window;
FUTURE = [];
%N_ITER = TOT_N - size(TX,1) - p + 1;
N_ITER = TOT_N - size(TX,1);
tic;
for iter=1:N_ITER
    Xts = window';
    step = net(Xts')'; 
    window = buffer(end-p+1:end);
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
%FUTURE=FUTURE(21:end); % Remove first training window
figure(2),clf, hold on;
plot(TOT_N-size(FUTURE,1)+1:TOT_N, TXF(TOT_N-size(FUTURE,1)+1:TOT_N));
plot(TOT_N-size(FUTURE,1)+1:TOT_N, FUTURE);
title('Mackey-Glass(30) Iterated Prediction', 'FontSize', 16);
xlabel('k', 'FontSize', 14)
ylabel('x', 'FontSize', 14);
legend('Mackey-Glass TS','FFNN Iterated Prediction','Location','southeast');
%xlim([size(TX,1),size(TXF,1)]);
%xlim([size(TX,1)-100,size(TXF,1)]);
%xlim([size(TXF,1)-500,size(TXF,1)]);
%xlim([size(TX,1)-100,size(TX,1)+100]);
%xlim([size(TXF,1)-200,size(TXF,1)]);


%RMSE
sqrt(mean((TXF(end-size(FUTURE,1)+1:end) - FUTURE).^2)), %RMSE

