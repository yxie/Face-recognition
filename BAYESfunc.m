function [solution] = BAYESfunc(D, DT, LT, c, delta)
%D: train set
%DT: test set
%LT: test label
%c: #classes
%delta: a constant that's used to make the covariance matrix non-singular

d = size(D,1); % # dimension
nt = size(DT,2); % # test data

mu = zeros(d, c); %mean of each class
for i=1:c
    mu(:, i) = (D(:,2*i-1) + D(:,2*i))/2;
end

var = zeros(d, d, c);
var_inv = zeros(d, d, c);
%delta = 2;
for i=1:c
   var(:, :, i) = 1/2 * ( D(:, 2*i-1) - mu(:,i) ) * ( D(:, 2*i-1) - mu(:,i) ).' ...
                + 1/2 * ( D(:, 2*i) - mu(:,i) ) * ( D(:, 2*i) - mu(:,i) ).'; 
   var(:, :, i) = var(:, :, i ) + delta * eye(d);
   if det(var(:,:,i)) == 0
      display('singular');
      pause;
   end
   var_inv(:, :, i) = inv(var(:, : ,i));
end


%testing
% max_prob = 0;
% solution = zeros(nt,1);
% for i=1:nt
%     for j=1:c
%         p = mvncdf(DT(:,i),mu(:,j),var(:,:,j));
%         if p > max_prob
%            max_prob = p;
%            solution(i) = j;
%         end
%     end
% end
% 
%calculate discriminant function
W = zeros(d, d, c);
w = zeros(d, c);
w0 = zeros(c,1);

for i = 1:c
    W(:,:,i) = -1/2 * var_inv(:, :, i);
    w(:,i) = var_inv(:, :, i) * mu(:,i);
    w0(i) = -1/2 * mu(:, i)' * var_inv(:, :, i) * mu(:, i) - 1/2 * log(det(var(:,:,i))); % ignore lnP(wi)
end

%Testing
solution = zeros(nt,1);
for i = 1:nt % test data DT
    display(i);
    max_g = -100000;
    for j = 1:c % discriminant functions
        g = DT(:,i)' * W(:,:,j) * DT(:,i) + w(:,j)' * DT(:,i) + w0(j);
        if(g > max_g)
           max_g = g; 
           solution(i) = j;
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

