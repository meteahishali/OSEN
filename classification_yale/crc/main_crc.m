clc, clear, close all

[Dic_all, train_all, test_all] = read_data();

nuR = 5; % Number of runs.
MR = 0.01;
rng(10)
acc = zeros(1, nuR);


[maskM, maskN] = size(Dic_all(1).label_matrix);
N = size(Dic_all(1).dictionary, 1);

for k = 1:nuR
    Dic = Dic_all(k);
    train = train_all(k);
    test = test_all(k);
    
    % Include all training samples to the dictionary.
    Dic.dictionary = [Dic.dictionary train.data];
    Dic.label =[Dic.label; train.label];
    D=Dic.dictionary; % This is the dictionary.
    
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

    % Projection matrix computing.
    kappa = [0.00001]; % l2 regularized parameter value.
    Proj_M = inv(A'*A+kappa*eye(size(A,2)))*A'; %l2 norm
        
    %%%% Testing with CRC.
    ID = [];
    test_length = size(Y2,2);
    tstart = tic;
    for indTest = 1:size(Y2,2)
        [id]    = CRC_RLS(A,Proj_M,Y2(:,indTest),Dic.label);
        ID      =   [ID id];
    end
    per.telapsed(1) = toc(tstart);
    cornum      =   sum(ID'==test.label);
 
    per.telapsed(1) = per.telapsed(1)./(test_length);
    Rec         =   [cornum/length(test.label)]; % Recognition rate.
    fprintf([' ' num2str(Rec)]);
    per.Rec(1, k) = Rec;
    
end

disp(strcat(' Averaged accuracy: ', num2str(mean(per.Rec))))