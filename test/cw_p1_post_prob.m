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

a = log(F2./F1);
POST = 1 ./ (1 + exp(-a));

POST = reshape(POST,length(x2),length(x1));
figure(1),clf
surf(x1,x2,POST);
%caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
%axis([-3 3 -3 3 0 .4])
xlabel('x1'); ylabel('x2'); zlabel('Posterior Probability');

%view([-1,-1,2])