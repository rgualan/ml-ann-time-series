m1 = [0; 3];
m2 = [2; 1];
C1 = [2 1; 1 2]; 
C2 = [1 0; 0 1]; 

%Limits
%x1Lim=[-4,6];x2Lim=[-4,6];
x1Lim = [-4.0 6.0];
x2Lim = [-2.0 6.0];

mu = m1';
%Sigma = [.25 .3; .3 1];
Sigma = C1;
x1 = x1Lim(1):.2:x1Lim(2); x2 = x2Lim(1):.2:x2Lim(2);
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
figure(1),clf
surf(x1,x2,F);
%caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
%axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
%return;

hold on
mu = m2';
%Sigma = [.25 .3; .3 1];
Sigma = C2;
x1 = x1Lim(1):.2:x1Lim(2); x2 = x2Lim(1):.2:x2Lim(2);
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
%axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
