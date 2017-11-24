% =========================================================================
% Distributions' feature
N=100; %100
m1 = [0; 3];
m2 = [2; 1];
C = [2 1; 1 2]; 
C1 = [2 1; 1 2]; 
C2 = [1 0; 0 1]; 
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);
X = [X1; X2];
y = [ ones(N,1); -1*ones(N,1) ];
%load('distributions.mat');


net = feedforwardnet(20);
net = train(net, X', y');
%view(net);
output = net(X');


% Plot decision boundary
x1Lim=[-4,6]; x2Lim=[-4,6];
x1 = x1Lim(1):.2:x1Lim(2); x2 = x2Lim(1):.2:x2Lim(2);
[Xg1,Xg2] = meshgrid(x1,x2);
OUT = net([Xg1(:) Xg2(:)]');
OUT2 = reshape(OUT,length(x2),length(x1));
OUT2 = OUT2>0;
figure(1),clf
hold on;
for i=1:size(OUT2,1)
    for j=1:size(OUT2,2)
        if OUT2(i,j) > 0
            plot(x1(i),x2(j),'b.');
        else
            plot(x1(i),x2(j),'r.');
        end
    end
end
xlabel('x1'); ylabel('x2');
xlim(x1Lim);ylim(x2Lim);




