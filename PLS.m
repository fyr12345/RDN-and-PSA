clc
clear
data=table2array(HCvsPT_ttresuluts);

X=zscore(data(:,1:830));%%FC
Y=zscore(data(:,831:848));%%AESitems
% Create an empty cell array to store the eligible Y-feature columns
selected_x_features = {};

% For each column in Y
for i = 1:size(X, 2)
    
    [correlation, pval] = corr(X(:, i), Y);
    
    % If the p-value of the correlation is less than xx, add the corresponding column of Y to the list of selected features
    % If the p-value of the correlation is less than xx, add the corresponding column of Y to the list of selected features,XX in the article is 1
    if any(pval < xx)
        selected_x_features{end+1} = X(:, i);
    end
end

% Combine the selected feature lists into one matrix
selected_x_features_matrix = cell2mat(selected_x_features);


% Find the index of the row containing nan in X 
nanRows = any(isnan(X),2); 
% Delete the corresponding rows in X and Y 
X(nanRows,:) = []; Y(nanRows,:) = [];
resmaple(nanRows,:) = [];
X=zscore(X);Y=zscore(Y);
resmaple=zscore(resmaple)
dim=10
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y,dim);
dim=5;
plot(1:dim,cumsum(100*PCTVAR(2,1:dim)),'-o','LineWidth',1.5,'Color',[140/255,0,0]);
set(gca,'Fontsize',14)
xlabel('Number of PLS components','FontSize',14);
ylabel('Percent Variance Explained in Y','FontSize',14);

dim=18;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(selected_x_features_matrix,Y,dim); % no need to do this but it keeps outputs tidy

%%% plot correlation of PLS component 1 with t-statistic (from Cobre as an example):
figure
plot(XS(:,5),YS(:,5),'r.')
[R,p]=corrcoef(XS(:,5),YS(:,5)) 
xlabel('XS scores for PLS component 1','FontSize',14);
ylabel('Cobre t-statistic- lh','FontSize',14);

%%%%%%%%%%%%%%permutation test 
%%%%%%%%%%%%%%permutation test 
rep=1000
R = zeros(1, 18);  % Initialise R to store Rsquared values for each dimension
p = zeros(1, 18);  % Initialise p for storing p values for each dimension



% Mess up 1000 times and keep a record of each mess up
num_permutations = 1000;
permutation_records = cell(num_permutations, 1);

for i = 1:num_permutations
    
    shuffled_index = randperm(size(Y, 1));
    
    
    shuffled_data = Y(shuffled_index, :);
    
    
    permutation_records{i} = shuffled_data;
end


dim=18
 for j=1:rep
    j
    
    Yp = permutation_records{j};
    [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(selected_x_features_matrix,Yp,dim);

    tem(j,:)=PCTVAR(2,:);
   
end



[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(selected_x_features_matrix,Y,dim);

for l=1:dim
    p_single(l)=length(find(tem(:,l)>=PCTVAR(2,l)))/1000;
end

figure
plot(1:dim, p_single,'ok','MarkerSize',8,'MarkerFaceColor','r');
xlabel('Number of PLS components','FontSize',14);
ylabel('p-value','FontSize',14);
grid on



