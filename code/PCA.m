clear all
close all
clc

% enter a path that contains all the vectors we want to
% find principal components for, in csv format
X = csvread('/home/tedz/visual_reps_100K.csv');

X = X';
mean_data = mean(X, 2);
% now we shift the data
mean_data = bsxfun(@minus,X,mean_data);
clear X; % free up the memory
% 
% 
YY = 1/size(mean_data, 1) * (mean_data * mean_data');
% this is a roundabout way of finding the covariance matrix
% advantage is it doesn't need to pre-load and store all 
% the numbers
% 
% uncomment the following if you want to see how much the
% each principal component contributes
% [v1,d1] = eig(YY);
% eigVals = diag(d1);
% sumDiag = sum(eigVals);
% eigValLargeSmall = eigVals(end:-1:1); %lists eigenvalues from largest to smallest
% k = 400 % plot the contribution of the first 400 principal components
% plot(cumsum(eigValLargeSmall(1:k))*100/sumDiag)


% reduce it to 200 principal components
pca = 200
[v, d] = eigs(YY, pca);
clear d;
% output file below
eigVecFile = 'visual_reps_reduced_100k_200.txt';
dlmwrite(eigVecFile,v);
time_passed = toc
