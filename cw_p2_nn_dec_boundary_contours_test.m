% =========================================================================
load('distributions.mat');
load('net-5.mat');
%N = 1000;
%X1 = mvnrnd(m1, C1, N); 
%X2 = mvnrnd(m2, C2, N);
%save('class_problem_testdata.mat','N', 'X1','X2');  
load('class_problem_testdata.mat');  


%BAYES - CLASSIFICATION
X = [X1; X2];
y = [ones(N,1); -1*ones(N,1)];

C1_inv = inv(C1);
C2_inv = inv(C2);
W1=(-1/2)*C1_inv;
W2=(-1/2)*C2_inv;
w1=C1_inv*m1;
w2=C2_inv*m2;
omega01=(-1/2)*m1'*C1_inv*m1 - (1/2)*log(det(C1)) + log(0.5);
omega02=(-1/2)*m2'*C2_inv*m2 - (1/2)*log(det(C2)) + log(0.5);
bayes = bayes_classifier(X, W1, W2, w1, w2, omega01, omega02 );

y2 = y./2+0.5;
bayes2 = bayes./2+0.5;
figure(1); 
gca = plotconfusion(y2', bayes2');
set(gca,'Position',[76 110 234 242]);
return;






%NEURAL NETWORK - CLASSIFICATION
X = [X1; X2];
y = [ ones(N,1); -1*ones(N,1) ];
output = net(X');

% Count missclassifications
E1 = 0; E2 = 0; LIMIT = 0;
for i=1:2*N
    if i<=N && output(i) < LIMIT 
        E1 = E1 + 1;
    end
    if i>N && output(i) > LIMIT
        E2 = E2 + 1;
    end
end
display(E1);display(E2);




% Confusion
y2 = y./2+0.5;
output2 = output'./2+0.5;
figure(1); 
gca = plotconfusion(y2', output2');
set(gca,'Position',[76 110 234 242]);

