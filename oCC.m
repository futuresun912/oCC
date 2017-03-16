function Pre_Labels = oCC(train_data,train_target,test_data,opts)
% oCC Optimized classifier chains for multi-label classification
%
%    Syntax
%
%       Pre_Labels = oCC(train_data,train_target,test_data,opts)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           opts             Parameters of oCC
%             opts.k         Percent of selected parents
%             opts.M         Percent of selected features
%
%       Output
%           Pre_Labels       An L x Nt output matrix, each column is a predicted label set

%% Get the problem size
[num_label,num_train] = size(train_target);
[num_test,num_feature] = size(test_data);

%% Set parameters
k = round( num_label * opts.k );
M = round( num_feature * opts.M );

%% Set the Liblinear package
if num_train > num_feature
    svmlinear = '-s 2 -B 1 -q';
else
    svmlinear = '-s 1 -B 1 -q';
end

%% Build a classifier chain in random label order
pa = [];
disc_train   = myDisc(train_data,3,0.5);
Pre_Labels   = zeros(num_test,num_label);    
null_target  = zeros(num_test,1); 
train_target = train_target';
train_data   = sparse(train_data);
test_data    = sparse(test_data);
% Chain order selection
chain = orderSelect(train_data,train_target,opts);
for j = chain 
    if ~all(train_target(:,j)==0)
        % Stage 1: label correlation modeling
        ind_L = oCC_select(k,train_target(:,pa),train_target(:,j));
        kpa = pa(ind_L);
        % Stage 2: multi-label feature selection
        ind_F = oCC_select(M,disc_train,train_target(:,j),train_target(:,kpa));
        % Stage 3: predictive model building
        model = libtrain(train_target(:,j),[train_data(:,ind_F),train_target(:,kpa)],svmlinear);
        Pre_Labels(:,j) = libpredict(null_target,[test_data(:,ind_F),Pre_Labels(:,kpa)],model,'-q');
        pa = [pa,j];
    end
end
Pre_Labels = Pre_Labels';

end