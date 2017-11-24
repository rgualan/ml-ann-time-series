m1 = [0; 3];
m2 = [2; 1];
C1 = [2 1; 1 2]; 
C2 = [1 0; 0 1]; 

x1Lim=[-4,6];
x2Lim=[-4,6];

x1 = x1Lim(1):.2:x1Lim(2); 
x2 = x2Lim(1):.2:x2Lim(2);
[X1,X2] = meshgrid(x1,x2);
F1 = mvnpdf([X1(:) X2(:)],m1',C1);
F2 = mvnpdf([X1(:) X2(:)],m2',C2);

a = log(F1./F2);
POST = 1 ./ (1 + exp(-a));
b = log(F2./F1);
POST2 = 1 ./ (1 + exp(-b));

POST = reshape(POST,length(x2),length(x1));
POST2 = reshape(POST2,length(x2),length(x1));
figure(1),clf
surf(x1,x2,POST);
%caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
%axis([-3 3 -3 3 0 .4])
%return;
title('Posterior Prob. for Class 2', 'FontSize', 16);
xlabel('x1', 'FontSize', 14)
ylabel('x2', 'FontSize', 14);
zlabel('Posterior Probability');


hold on
surf(x1,x2,POST2);
