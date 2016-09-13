clc;
clear; 
load('./data/data.mat'); % 24 * 21 * 600 ?200 objects, each has 3 pics)
d = 24*21; % # dimension
c = 200; % # class
n = 400; % # training data
nt = 200; % # test data
D = zeros(d, n); % Training data set 504 * 400 (first 2 pics of each object)
DT = zeros(d, nt); % Test data set 504 * 200 (last pics of each object)
L = zeros(n,1); % label for training data
LT = zeros(nt,1); % label for test data
for i=0:c-1
    count = 1;
    for j=1:3
        if j==2 || j==3
            D(:,2*i+count)=reshape(face(:,:,3*i+j), [d,1]);
            L(2*i+count) = i+1; 
            count = count + 1;
        else
            DT(:,i+1)=reshape(face(:,:,3*i+j), [d,1]);            
            LT(i+1) = i+1;
        end
    end
end

[W,S,V] = svds(D,c-1);


Y = zeros(c-1, n);
YT = zeros(c-1, nt);
for i = 1:n
   Y(:, i) = W.' * D(:,i);
end
for i = 1:nt
    YT(:, i) = W.' * DT(:,i);
end

delta = 1; %var singularity
solution = BAYESfunc(Y, YT, LT, c, delta);

%train 1,2 test 3
%accuracy = 0.635

%train 1,3 test 2
%accuracy = 0.66

%train 2,3 test 1
%accuracy = 0.71
