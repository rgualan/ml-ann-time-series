%[ ~, TX ] = mackeyglass_func(2000);
[ ~, TX ] = mackeyglass_func(2000, 0.2, 0.1, 17, 1.2, 1);

TX = TX(1:2000); %plot(TX); return;
TX1 = TX(1:1500); TX2 = TX(1501:2000);
p = 20;
%plot(TX); return;


%Set up training and test data
%N = size(TX,1);
%Ntr = N-(p+1);
Ntr = size(TX1,1)-p;
Nts = size(TX2,1)-p;
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
net = feedforwardnet(20);
net = train(net, Xtr', ytr');
yhtr = net(Xtr')'; % test
yhts = net(Xts')'; % test


%Ploting
figure(1),clf;
plot(TX, 'LineWidth', 2); hold on; 
plot(size(TX,1)-size(yhts,1)+1:size(TX,1),yhts,'r', 'LineWidth', 2);
title('TS prediction - Mackey-Glass model', 'FontSize', 16);
legend('Original','Predicted', 'Location','southeast');

%Max Error
%max(abs(yhts-yts)),
sqrt(mean((yts - yhts).^2)), %RMSE
pred2 = yhts;


%Ploting - compair
figure(1),clf;
plot(TX, 'LineWidth', 2); hold on; 
%pred1: cvx tool
%pred2: ann
plot(size(TX,1)-size(yhts,1)+1:size(TX,1),pred1, 'LineWidth', 2);
plot(size(TX,1)-size(yhts,1)+1:size(TX,1),pred2, 'LineWidth', 2);
title('TS prediction - Mackey-Glass model', 'FontSize', 16);
xlabel('n', 'FontSize', 14)
ylabel('s(n)', 'FontSize', 14);
legend('Original','Linear predictor (CVX Tool)', 'Feedforward NN', 'Location','southeast');

