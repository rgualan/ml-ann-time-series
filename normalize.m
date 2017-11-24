function [ XN ] = normalize( X )
    XN = X - mean(X);
    XN = XN/std(X);
end

