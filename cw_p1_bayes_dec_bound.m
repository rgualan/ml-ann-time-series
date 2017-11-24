% =========================================================================
% Distributions' feature
N=100;
m1 = [0; 3];
m2 = [2; 1];
%m2 = m2 + [5; 0];
C1 = [2 1; 1 2]; 
C2 = [1 0; 0 1]; 
%X1 = mvnrnd(m1, C1, N); X2 = mvnrnd(m2, C2, N);
load('distributions.mat');


%Limits
x1Lim = [-3.0 5.0];
x2Lim = [-2.0 6.0];


%Probability contour plot
numGrid = 50;
xRange = linspace(x1Lim(1), x1Lim(2), numGrid);
yRange = linspace(x2Lim(1), x2Lim(2), numGrid);
P1 = zeros(numGrid, numGrid);
P2 = zeros(numGrid, numGrid);
for j=1:numGrid
    for i=1:numGrid;
        x = [xRange(j) yRange(i)]';
        P1(i,j) = mvnpdf(x', m1', C1);
        P2(i,j) = mvnpdf(x', m2', C2);
    end
end
Pmax = max(max([P1 P2]));
figure(1), clf,
contour(xRange, yRange, P1, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 1);
hold on;
contour(xRange, yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 1);
% Plot centres
plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
plot(m2(1), m2(2), 'r*', 'LineWidth', 4);
% Plot distributions
plot(X1(:,1),X1(:,2),'bx', X2(:,1),X2(:,2),'ro'); grid on;
title('Bayesian quadratic decision boundary', 'FontSize', 16);
xlabel('x1', 'FontSize', 14)
ylabel('x2', 'FontSize', 14);
%return;


% Plot quadratic classification boundary
% gi(x) = x^t W_i x + w^t_i x + w_i0,   
C1_inv = inv(C1);
C2_inv = inv(C2);
W1=(-1/2)*C1_inv;
W2=(-1/2)*C2_inv;
w1=C1_inv*m1;
w2=C2_inv*m2;
omega01=(-1/2)*m1'*C1_inv*m1 - (1/2)*log(det(C1)) + log(0.5);
omega02=(-1/2)*m2'*C2_inv*m2 - (1/2)*log(det(C2)) + log(0.5);

%Solve
x = sym('x',[2 1]); %x = sym(x,'real');
assume(x,'real');

g1= x'*W1*x + w1'*x + omega01;
g2= x'*W2*x + w2'*x + omega02;

%Plot solution
[equation, ~, conds] = solve(g1 == g2, x(2,1), 'ReturnConditions', true);
xi=linspace(-0.6373,6,numGrid);
yi=zeros(2,numGrid);
for i=1:numGrid
    x1 = xi(i);
    yi(:,i) = subs(equation);
end
%figure(2), clf
plot(xi,yi(1,:),'r', 'LineWidth', 4); 
plot(xi,yi(2,:),'r', 'LineWidth', 4);
xlim(x1Lim);
ylim(x2Lim);




% Classify data
X = [X1; X2];
y = [ones(N,1); -1*ones(N,1)];
bayes = bayes_classifier(X, W1, W2, w1, w2, omega01, omega02 );

y2 = y./2+0.5;
bayes2 = bayes./2+0.5;
figure(2); 
gca = plotconfusion(y2', bayes2');
set(gca,'Position',[76 110 234 242]);


E1=0;E2=0;LIMIT=0;

for i=1:2*N
    % Distribution 1
    if i<=N && bayes(i) < LIMIT 
        E1 = E1 + 1;
    end
    % Distribution 2
    if i>N && bayes(i) > LIMIT
        E2 = E2 + 1;
    end
end
display(E1); display(E2);


%save('distributions.mat','N', 'C1', 'C2',  'm1', 'm2', 'X1','X2');  % function form