clc, clear, close all

addpath(genpath('l1benchmark'));
addpath(genpath('l1magic-1'));
addpath(genpath('gpsr'));

% Supported classifiers.
%l1method={'solve_ADMM', 'solve_dalm', 'solve_OMP', 'solve_homotopy', 'solveGPSR_BCm', 'solve_L1LS', 'solve_l1magic', 'solve_PALM'}; %'solve_PALM' is very slow
%l1method_names={'ADMM', 'DALM', 'OMP', 'Homotopy', 'GPSR', 'L1LS', 'l1magic', 'PALM'};
%'solve_PALM' is very slow

l1method={'solve_ADMM'}; 
l1method_names={'ADMM'};
    
[Dic_all, train_all, test_all] = read_data();

nuR = 5; % Number of runs.
MR = 0.01;
rng(10)
acc = zeros(1, nuR);


[maskM, maskN] = size(Dic_all(1).label_matrix);
N = size(Dic_all(1).dictionary, 1);

for k = 1:nuR
    fprintf(['Run ' num2str(k) '\n']);
    Dic = Dic_all(k);
    train = train_all(k);
    test = test_all(k);
    
    % Include all training samples to the dictionary.
    Dic.dictionary = [Dic.dictionary train.data];
    Dic.label =[Dic.label; train.label];
    D=Dic.dictionary; % This is the dictionary.
    
    Noise = 10;
    
    m = floor(MR * N); % Number of measurements.
    
    % Eigenface extracting.
    [phi,disc_value,Mean_Image]  =  Eigenface_f(D,m);
    phi = phi';
        
    A = phi*D;
    A = A./( repmat(sqrt(sum(A.*A)), [m,1]) ); % Normalization.

    % Measurements for test set.
    Y2= phi*test.data;
    energ_of_Y2=sum(Y2.*Y2);
    tmp=find(energ_of_Y2==0);
    Y2(:,tmp)=[];
    test.label(tmp)=[];
    test.data(:, tmp)=[];
    Y2 = Y2./( repmat(sqrt(sum(Y2.*Y2)), [m,1]) ); % Normalization.
        
    test_length=length(test.label);
    
    for i=1:length(l1method)
        fprintf(l1method_names{i});
        ID = [];
        tstart = tic;
        for indTest = 1:test_length
            [id]    =  L1_Classifier(A,Y2(:,indTest),Dic.label,l1method{i});
            ID      =   [ID id];
        end
        per.telapsed(i+1, k) = toc(tstart);
        per.telapsed(i+1, k) = per.telapsed(i+1, k)./(test_length);
        cornum      =   sum(ID'==test.label(1:test_length));
        per.Rec(i, k)         =   [cornum/test_length]; % recognition rate
        fprintf([' Accuracy: ' num2str(per.Rec(i, k)) '\n']);
    end
end

for i = 1:length(l1method)
    disp(l1method_names{i})
    mean(per.Rec(i, :))
end