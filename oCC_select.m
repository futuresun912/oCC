function idx_select = oCC_select(K, data, target, parent)
% oCC_select Information theoretic model/feature selection for oCC 
%
%    Syntax
%
%       idx_select = oCC_select(K, data, target, parent)
%
%    Description
%
%       Input:
%           K           The number of selected features
%           data        An N x d matrix, each row is a data set
%           target      An N x 1 array, each element is a target label
%           parent      An N x |Pa| matrix, each row is parent set
%
%       Output
%           idx_select  A K x 1 array, containing the indcies of selected variables


%% Set parameters
if nargin == 4;
    beta  = 1;
    gamma = 1;
elseif nargin == 3
    beta  = 0;
    gamma = 1;
else
    error('Wrong input parameters, please type "help oCC_select" for more information');
end

%% Check if K is enough smaller
num_fea = size(data,2);
if num_fea <= K
    idx_select = 1:num_fea;
    disp('No selection is performed');
    return
end

%% Cache results on the relevancy term
rel = zeros(1,num_fea);
for i = 1:num_fea
    rel(i) = mi(data(:,i), target);
end

%% Restrict the search space by MAX
MAX = 200;
idx_select = zeros(K,1);
[~, idxs] = sort(rel,'descend');
if MAX <= K
    idx_select = idxs(1:MAX);
    return;
else
    idx_select(1) = idxs(1);
    KMAX = min(MAX,num_fea);
    idxleft = idxs(2:KMAX);
    num_select = 1;
    num_unselect = KMAX-1;
end

%% Cache the correlation term
if beta == 0
    correl = zeros(1,KMAX);
else
    num_parent = size(parent,2);
    if num_parent > 0
        cor_matrix = zeros(num_parent,KMAX);
        for i = 1:KMAX
            if i == 1
                cor_matrix(:,i) = zeros(num_parent,1);
            else
                for j = 1:num_parent
                    cor_matrix(j,i) = cmi(target,parent(:,j),data(:,idxs(i)));
                end
            end
        end
        correl = mean(cor_matrix,1);
    else
        correl = zeros(1,KMAX);
    end
end

%% Greedy iterative optimization
diff_mat = zeros(num_unselect,K);
for k = 2:K
    rel_mi = zeros(num_unselect,1);   % Relevance
    cor_mi = zeros(num_unselect,1);   % Label Correlation
    diff_mi = zeros(num_unselect,1);  % Redundancy Difference
    for i = 1:num_unselect,
        rel_mi(i) = rel(idxleft(i));
        cor_mi(i) = correl(idxs==idxleft(i));
        diff_mat(idxleft(i),num_select) = cmi(target, data(:,idx_select(num_select)),...
            data(:,idxleft(i)));
        diff_mi(i) = mean(diff_mat(idxleft(i), 1:num_select));
    end
    
    % Information theoretic selection criterion
    [~, idx_select(k)] = max( rel_mi + beta*cor_mi + gamma*diff_mi);
    
    tmpidx = idx_select(k);
    idx_select(k) = idxleft(tmpidx);
    idxleft(tmpidx) = [];
    num_select = num_select + 1;
    num_unselect = num_unselect - 1;
end

end