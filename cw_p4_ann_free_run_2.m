SUB_N = 1000;
[ ~, TX ] = mackeyglass_func(SUB_N);
TX = TX(1:SUB_N);
TX = normalize(TX);
p = 20;


%Set up training data
Ntr = size(TX,1)-p;
Xtr = zeros(Ntr,p); ytr = zeros(Ntr,1); 
for i=1:Ntr
    Xtr(i,:) = TX(i:i+p-1)';
    ytr(i) = TX(i+p);
end
net = feedforwardnet(20);
net = train(net, Xtr', ytr');
yhtr = net(Xtr')';


%Ploting
%figure(1),clf;
%plot(TX, 'LineWidth', 2); hold on; 
%plot(size(TX,1)-size(yhtr,1)+1:size(TX,1),yhtr,'r--');
max(abs(yhtr-ytr)), %Max Error
%return;


TOT_N = 20000; 
[ ~, TXF ] = mackeyglass_func(TOT_N);
TXF = TXF(1:TOT_N);
TXF = normalize(TXF);
plot(TXF); hold on;
%plot(TXF(1:2000),'r');
%return;




%A WINDOW at a time
window = yhtr;
buffer = window;
FUTURE = buffer;
N_ITER = floor( TOT_N/size(window,1) );
tic;
for iter=1:N_ITER
    Nts = size(window,1)-p;
    Xts = zeros(Nts,p);
    for i=1:Nts
        Xts(i,:) = window(i:i+p-1)';
    end

    window = net(Xts')'; 
    buffer = [buffer; window];
    FUTURE = [FUTURE; window];
    
    window = buffer(end-size(window,1)+1:end);
    %stop condition
    if range(window) < 0.0001
        disp('STOP CONDITION FIRED!')
        break;
    end
    
    %reduce buffer size
    buffer = buffer(end-size(window,1)+1:end);    
end
toc;
plot(TOT_N-size(FUTURE,1)+1:TOT_N, FUTURE);






