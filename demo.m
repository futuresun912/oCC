% This is an example program for the paper: 
% 
% L. Sun and M. Kudo. Optimization of Classificer Chains via Conditional Likelihood Maximization. 
% A submission to Pattern Recognition. 
%
% The program shows how the oCC/EoCC program (The main function is 'oCC.m'/'EoCC.m') can be used.
%
% Please type 'help oCC' or 'help EoCC' under MATLAB prompt for more information.
%
% The program was developed based on the following packages:
%
% [1] Liblinear
% URL: https://www.csie.ntu.edu.tw/~cjlin/liblinear/
%
% [2] MIToolbox
% URL: http://www.cs.man.ac.uk/~pococka4/MIToolbox.html
% 
% The mex files of Liblinear and MIToolbox is generated in Windows(64bit).
% If you want to conduct the program in other systems, please compile relevant
% C files of the packages as follows:
% 1. Type 'cd ..\oCC\func\liblinear', and then 'make';
% 2. Type 'cd ..\oCC\func\MIToolbox', and then 'CompileMIToolbox'.


%% To repeat the experiments
rng('default');

%% Add necessary pathes
addpath('data','eval');
addpath(genpath('func'));

%% Set parameters
% oCC
occ.k    = 0.8;      % percent of selected parents
occ.M    = 0.8;      % percent of selected features
% For chain order selection
occ.dim  = 1;        % dimensionality of features 
occ.kmax = 0;        % maximun number of parents
occ.alg  = 'cos';    % chain order selection
% EoCC
eocc.m   = 10;       % number of ensembles
eocc.k   = 0.8;      % percent of selected parents
eocc.M   = 0.8;      % percent of selected features
eocc.per_F = 0.5;    % random sampling on features
eocc.per_N = 0.75;   % random sampling on instances
eocc.alg = 'random'; % random order selection 

%% Choose a datasets and method
dataset = 'scene';
method  = 'occ';
load([dataset,'.mat']);

%% Perform n-fold cross validation
num_fold = 5;
Results  = zeros(5,num_fold);
indices  = crossvalind('Kfold',size(data,1),num_fold);
for i = 1:num_fold
    disp(['Round ',num2str(i)]);
    test = (indices == i); train = ~test;
    switch method
        case 'occ'
            tic; Pre_Labels = oCC(data(train,:),target(:,train),data(test,:),occ);
        case 'eocc'
            tic; Pre_Labels = EoCC(data(train,:),target(:,train),data(test,:),eocc);
    end
    Results(1,i) = toc;
    Results(2:end,i) = Evaluation(Pre_Labels,target(:,test));
end
meanResults = squeeze(mean(Results,2));
stdResults  = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],[dataset,'_',method],...
    'Time ExactM HammingS MacroF1 MicroF1','Mean Std.');