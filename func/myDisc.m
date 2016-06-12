function train_data = myDisc( train_data,num_state,factor )
%MYDIST Discretize the training data for computing mutual information

num_feature = size(train_data,2);
if num_state == 3
    values = [-1,0,1];
elseif num_state == 5
    values = [-2,-1,0,1,2];
else
    disp('#State for discretization is set as 3.')
    num_state = 3;
end
for i = 1 : num_feature
    array_fea = train_data(:,i);
%     if max(array_fea) ~= 1
    if  max(array_fea(array_fea~=1)) ~= 0
        meanF = mean(array_fea);
        stdF1 = factor*std(array_fea);
        if num_state == 3
            edges = [-inf; meanF-stdF1; meanF+stdF1; inf];
        else
            stdF2 = 2*stdF1;
            edges = [-inf; meanF-stdF2; meanF-stdF1; meanF+stdF1; meanF+stdF2; inf];
        end
        edges = round(edges,3);
%         edges = round(edges);
        train_data(:,i) = discretize(array_fea,edges,values);
    end
end

end