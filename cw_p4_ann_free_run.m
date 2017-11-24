%TOT_N = 5000; %Total number of rows of the dataset
TOT_N = 2000;
%[ ~, TXF ] = mackeyglass_func(TOT_N);
[ ~, TXF ] = mackeyglass_func(TOT_N, 0.2, 0.1, 17, 1.2, 1); 

TXF = TXF(1:TOT_N);
%TXF = normalize(TXF);
p = 20;
figure(1),clf; plot(TXF); hold on;
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

%Ploting
% figure(2),clf;
% plot(TX, 'LineWidth', 2); hold on; 
% plot(size(TX,1)-size(yhtr,1)+1:size(TX,1),yhtr, 'LineWidth', 2);
% max(abs(yhts-yts)), %Max Error
% return;




% Free running mode
% %yhts = yhts;
% BIG_Y = yhts;
% ws = size(yhts,1); % window size
% for iter=1:1000
%     if size(yhts,1)-p <= 0
%         disp('No more data at iteration');
%         disp(iter);
%         break;
%     end
%     
%     Nts = size(yhts,1)-p;
%     Xts = zeros(Nts,p);
%     for i=1:Nts
%         Xts(i,:) = yhts(i:i+p-1)';
%     end
%     yhts = net(Xts')'; 
%     BIG_Y = [BIG_Y; yhts];
%     yhts = BIG_Y(end-ws+1:end);
% end
% figure(2), clf;
% plot(BIG_Y);


% Free running mode (NO storage of historical data)
% %yhts = yhts;
% buffer = yhts;
% ws = size(yhts,1); % window size
% N_ITER = 1000000;
% %ranges= zeros(N_ITER);
% tic;
% for iter=1:N_ITER
%     Nts = size(yhts,1)-p;
%     Xts = zeros(Nts,p);
%     for i=1:Nts
%         Xts(i,:) = yhts(i:i+p-1)';
%     end
%     yhts = net(Xts')'; 
%     buffer = [buffer; yhts];
%     yhts = buffer(end-ws+1:end);
%         
%     %stop condition
%     %ranges(iter) = range(yhts);
%     if range(yhts) < 0.0001
%         disp('STOP CONDITION FIRED!')
%         break;
%     end
%     
%     %reduce buffer size
%     buffer = buffer(end-ws-10:end);
% end
% toc;
% figure(2), clf, plot(buffer);
% %figure(3), clf, plot(ranges);




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
%legend('Mackey-Glass Series','FFNN Iterated Prediction','Location','southeast');
%xlim([size(TX,1),size(TXF,1)]);
%xlim([size(TX,1)-100,size(TXF,1)]);
%xlim([size(TXF,1)-500,size(TXF,1)]);
%xlim([size(TX,1)-100,size(TX,1)+100]);
%xlim([size(TXF,1)-200,size(TXF,1)]);


%RMSE
sqrt(mean((TXF(end-size(FUTURE,1)+1:end) - FUTURE).^2)), %RMSE



