function [solution] = NNfunc(D, DT, L, LT) %train data, test dat, train label, test label, #class
%load(filename); % 24 * 21 * 600 ?200 objects, each has 3 pics)
%d = size(D,1); % # dimension
n = size(D,2); % # training data
nt = size(DT,2); % # test data


solution = zeros(nt,1);
for i=1:nt
   min_dist = 100000;
   for j=1:n
      if (DT(:,i) - D(:,j))'*(DT(:,i) - D(:,j)) < min_dist
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

