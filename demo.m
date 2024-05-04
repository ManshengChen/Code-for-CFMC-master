warning off;
addpath(genpath('./'));

%% dataset
ds = {'Handwritten_fea'};
dsPath = './0-dataset/';
resPath = './res/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};

for dsi = 1:length(ds)
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    for i=1:6
        X{i}=X{i}';
    end
    k = length(unique(Y));
    
    matpath = strcat(resPath,dataName);
    txtpath = strcat(resPath,strcat(dataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    %% para setting
    anchor = [k];
    iters = 10;
    
    %%
    for ichor = 1:length(anchor)
        for iter=1:iters
            tic;
            [A,W,Ypre,alpha,obj,Yloss] = Yvectest(X,Y,anchor(ichor));
            timer  = toc;
            res = Clustering8Measure(Y,Ypre); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
            loss = obj(end);
            save(['res/', dataName, '_OCF_res_', num2str(anchor(ichor)), '_iter_', num2str(iter), '.mat'], 'dataName', 'A', 'W', 'Ypre', 'alpha', 'res', 'obj', 'timer', 'loss','Yloss');
            fprintf('\niter: %d, time: %.2f', iter, timer);
        end
        % get res (corresponding to the minimal loss) 
        vals = cell(iters, 1);
        tses = zeros(iters, 1);
        losses = zeros(iters, 1);

        for iter=1:iters
        load(['res/', dataName, '_OCF_res_', num2str(anchor(ichor)), '_iter_', num2str(iter), '.mat'])
        vals{iter} = res';
        tses(iter) = timer;
        losses(iter) = loss;
        end

        [~, ind] = min(losses);
        fprintf('\nsel.. Anchor:%d\t, loss: %.4f, acc: %.4f, nmi: %.4f, pur: %.4f, fscore: %.4f, ts: %.2f', anchor(ichor), losses(ind), vals{ind}(1), vals{ind}(2), vals{ind}(3), vals{ind}(4), tses(ind));
    end
    clear resall objall X Y k
end