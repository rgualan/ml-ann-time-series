function [ y ] = bayes_classifier(X, W1, W2, w1, w2, omega01, omega02 )

    y = zeros(size(X,1),1);

    for i=1:size(X,1)
        x = X(i,:)';
        g1= x'*W1*x + w1'*x + omega01;
        g2= x'*W2*x + w2'*x + omega02;
        if g1>g2
            y(i) = 1;
        else
            y(i) = -1;
        end
    end


end

