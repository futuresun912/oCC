function Pre_Labels = EoCC(train_data,train_target,test_data,opts)
%ECC Ensemble of Optimized Classifier Chains
%
%    Syntax
%
%       [Pre_Labels,Outputs] = EoCC(train_data,train_target,test_data,m)
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           opts             Parameters of EoCC
%             opts.m         Number of ensembles
%             eocc.per_F     Percent of features after random selection
%             eocc.per_N     Percent of instances after random selection
%             opts.k         Percent of selected parents, for each model
%             opts.M         Percent of selected features, for each model
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set

%% Set parameters
m       =   opts.m;
fea_per =   opts.per_F;
ins_per =   opts.per_N;

%% Set the size of sub-problem
[num_ins,num_fea] = size(train_data);
D = round(num_fea*ins_per);
N = round(num_ins*fea_per);

%% Build the ensemble of classifiers
Outputs = zeros(size(train_target,1),size(test_data,1));
for i = 1:m
    % generate the random number
    idx_fea = randperm(num_fea);
    idx_ins = randperm(num_ins);
    % Find the subsets of features and instances
    sub_train  = train_data(idx_ins(1:N),idx_fea(1:D));
    sub_target = train_target(:,idx_ins(1:N));
    sub_test   = test_data(:,idx_fea(1:D));
    % Build the classifiers in the subspace
    Temp_Labels = oCC(sub_train,sub_target,sub_test,opts);
    Outputs = Outputs + Temp_Labels;
end

Outputs = Outputs ./ m;
Pre_Labels = round(Outputs);

end