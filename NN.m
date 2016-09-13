clc;
clear; 

dataset = 3;

if dataset == 1
    load('./data/data.mat'); % 24 * 21 * 600 ?200 objects, each has 3 pics)
    d = 24*21; % # dimension
    c = 200; % # class
    ni = 2; % # training data per subject
    n = 400; % # training data
    nt = 200; % # test data
    D = zeros(d, n); % Training data set 504 * 400 (first 2 pics of each object)
    DT = zeros(d, nt); % Test data set 504 * 200 (last pics of each object)
    L = zeros(n,1); % label for training data
    LT = zeros(nt,1); % label for test data

    for i=0:c-1
        count = 1;
        for j=1:3
            if j==1 || j==2 %control training set
                D(:,2*i+count)=reshape(face(:,:,3*i+j), [d,1]);
                L(2*i+count) = i+1; 
                count = count + 1;
            else
                DT(:,i+1)=reshape(face(:,:,3*i+j), [d,1]);
                LT(i+1) = i+1;
            end
        end
    end
elseif dataset == 2
    load('./data/pose.mat');  %48*40*13*68, 13 poses, 68 subjects
    d = 48*40; % # dimension
    c = 68; % # class
    ratio = 0.1; % percentage of data for training
    ni = round(13*ratio) % # training data per subject
    n = ni*68 % # training data
    nt = (13-ni)*68 % # test data
    D = zeros(d,n);
    L = zeros(n,1);
    DT = zeros(d,nt);
    LT = zeros(nt,1);
    
    for i=0:c-1
        for j=1:ni
            D(:, ni*i+j) = reshape(pose(:,:,j,i+1), [d,1]);
            L(ni*i+j) = i+1;
        end
        for j=1:13-ni
            DT(:, (13-ni)*i+j) = reshape(pose(:,:,ni+j,i+1), [d,1]);
            LT((13-ni)*i+j) = i+1; 
        end
    end
else
    load('./data/illumination.mat');  %1920*21*68, 21 illuminations, 68 subjects
    d = 1920; % # dimension
    c = 68; % # class
    ratio = 0.5; % percentage of data for training
    ni = round(21*ratio) % # training data per subject
    n = ni*68 % # training data
    nt = (21-ni)*68 % # test data
    D = zeros(d,n);
    L = zeros(n,1);
    DT = zeros(d,nt);
    LT = zeros(nt,1);
    
    for i=0:c-1
        for j=1:ni
            D(:, ni*i+j) = reshape(illum(:,j,i+1), [d,1]);
            L(ni*i+j) = i+1;
        end
        for j=1:21-ni
            DT(:, (21-ni)*i+j) = reshape(illum(:,ni+j,i+1), [d,1]);
            LT((21-ni)*i+j) = i+1; 
        end
    end
end

solution = zeros(nt,1);
for i=1:nt
   min_dist = (DT(:,i) - D(:,1))'*(DT(:,i) - D(:,1));
   for j=1:n
      if (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j)) <= min_dist
         min_dist =  (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j));
         solution(i) = L(j);
      end
   end
end

accuracy = 0.0;
for i=1:nt
   if solution(i) == LT(i)
       accuracy = accuracy + 1;
   end
end
accuracy = accuracy / nt;
display(accuracy);

%result
%train:1,2; test:3
%accuracy = 0.595

%train:1,3; test:2
%accuracy = 0.65

%train:2,3; test:1
%accuracy = 0.555

