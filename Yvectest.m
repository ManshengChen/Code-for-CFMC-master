function [A,W,Y,alpha,obj,loss] = Yvectest(X,gt,d)
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

k = length(unique(gt));
m = length(X);
n = size(gt,1);

%Initilize Y
Y = randi([1,k],n,1);

A = cell(m,1); 
W = cell(m,1); 
for i = 1:m
   di = size(X{i},2); 
   A{i} = rand(di,d); % di * d
   W{i} = rand(d,k); % d * k
   X{i} = mapstd(X{i}',0,1); % turn into dv*n
end

alpha = ones(1,m)/m;
opt.disp = 0;

flag = 1; %judge if convergence
iter = 0;
%%
while flag
    iter = iter + 1;
    XvYT = cell(m,1);
    %% optimize A_i  dv*d
    for iv=1:m
        XvYT{iv} = zeros(size(X{iv},1),k);
        for j=1:k
            XvYT{iv}(:,j) = sum(X{iv}(:,Y==j), 2);
        end
        [Ua,~,Va] = svds(XvYT{iv}*W{iv}',d);
        A{iv} = Ua*Va';
    end
    
    %% optimize W_i d*k
    %=====W_i>=0
    for iw = 1:m
        YYTvec = zeros(k,1);
        for j=1:k
            YYTvec(j) = sum(Y==j);
        end
        YYTplusIinvvec = 1./(YYTvec+eps);
        tmp = A{iw}'*XvYT{iw};
        W{iw} = (repmat(YYTplusIinvvec, 1, d).*tmp')'; %copy n of 1*k
        W{iw}(W{iw}<0)=0;
    end
    
    %% optimize Y  k*n
    loss = zeros(n, k);
    for ij=1:m
        loss = loss + alpha(ij) * EuDist2(X{ij}', W{ij}'*A{ij}', 0);
    end
    [~, Y] = min(loss, [], 2); %minvalue of row,minvalue_position
    

    %% optimize alpha
    aloss = zeros(1,m);
    for iv = 1:m
        aloss(iv) = sqrt(sum(sum((X{iv}-A{iv}*W{iv}(:,Y)).^2)));
    end
    alpha=1./(2*aloss);

    %%
    term = zeros(m, 1);
    for iv = 1:m
        term(iv) = sum(sum((X{iv}-A{iv}*W{iv}(:,Y)).^2));
    end
    obj(iter) = alpha*term;
    
    
    if (iter>2) && (abs((obj(iter)-obj(iter-1))/(obj(iter)))<1e-5 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end
         
         
    
