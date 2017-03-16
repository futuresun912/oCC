function [ pi ] = orderSelect( data, target, opts )
% Selection of chain order (pi) by greedy forward search 
% data     N x d feature matrix, each row is an instance
% target   N x q label matrix, each row is a label set

%% Initialization
alg = opts.alg;
q   = size(target,2);
switch alg
    case 'cos'
        dim   = opts.dim;
        kMAX  = opts.kmax;
    case 'random'
        pi = randperm(q);
        return
end

%% Reduce the dimensionality of data and discretize it
data  = bsxfun(@minus,data,mean(data,1));
[V,~] = eigs(data'*data,dim);
data  = data * V;
data  = myDisc(data,3,0.5);

%% Cache the MI matrix
temp_CMI = zeros(q,q,dim);
for i = 1 : dim
    for j = 1 : q
        for k = (j+1) : q
            temp_CMI(j,k,i) = cmi(target(:,j),target(:,k),data(:,i));
        end
    end
end
temp_CMI = temp_CMI + permute(temp_CMI,[2,1,3]);
mat_CMI  = sum(temp_CMI,3) / dim;

%% Initialize the root and S_pi
pi = zeros(1,q);
[~,pi(1)] = max(sum(mat_CMI,1));
pi_select = pi(1);
pi_unselect = 1:q;
pi_unselect(pi_unselect==pi(1)) = []; 
num_select = 1;
mat_CMI(:,pi(1)) = 0;

%% Greedy iterative algorithm
for j = 2 : q-1
    rel_mat = mat_CMI(pi_select,:);    
    red_mat = zeros(num_select,q);
    if (j > 2) && (kMAX > 0)
        for pi_now = pi_unselect
            for k = 2 : j-1
                red_array = zeros(k-1,1);
                if (k-1) <= kMAX
                    l_start = 1;
                else 
                    l_start = k-kMAX;
                end
                for l = l_start : k-1
                    red_array(l)  = cmi(target(:,pi(k)),target(:,pi(l)),target(:,pi_now));
                end
                red_mat(k,pi_now) = sum(red_array) / (k-1);
            end
        end
    end
    [~, pi_j] = max(sum(rel_mat+red_mat,1));
    pi_select = [pi_select,pi_j];
    pi_unselect(pi_unselect==pi_j) = [];
    mat_CMI(:,pi_j) = 0;
    pi(j) = pi_j;
    num_select = num_select + 1;
end
pi(q) = pi_unselect(1);

end