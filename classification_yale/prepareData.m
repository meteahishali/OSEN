clc, clear, close all

addpath(genpath('crc'));

outFolder = 'OSENData/';
if ~exist(outFolder, 'dir')
    mkdir(outFolder)
end

[Dic_all, train_all, test_all] = read_data();

nuR = 5; % Number of runs.
MRs = [0.01, 0.05, 0.25];
for mr = 1:3
    MR = MRs(mr);
    disp(strcat('MR: ', num2str(MR)));
    rng(10)
    acc = zeros(1, nuR);

    [maskM, maskN] = size(Dic_all(1).label_matrix);
    N = size(Dic_all(1).dictionary, 1);

    for k = 1:nuR
        Dic = Dic_all(k);
        train = train_all(k);
        test = test_all(k);

        D=Dic.dictionary; % This is the dictionary.

        Noise = 10;

        m = floor(MR * N); % Number of measurements.

        % Eigenface extracting.
        [phi,disc_value,Mean_Image]  =  Eigenface_f(D,m);
        phi = phi';

        A = phi*D;
        A = A./( repmat(sqrt(sum(A.*A)), [m,1]) ); % Normalization.

        % Measurements for dictionary.
        Y0 = phi * Dic.dictionary;
        energ_of_Y0 = sum(Y0.*Y0);
        tmp = find(energ_of_Y0 == 0);
        Y0(:,tmp)=[];
        train.label(tmp) = [];
        Y0 = Y0./( repmat(sqrt(sum(Y0.*Y0)), [m,1]) ); % Normalization.

        % Measurements for training set.
        Y1= phi*train.data;
        energ_of_Y1=sum(Y1.*Y1);
        tmp=find(energ_of_Y1==0);
        Y1(:,tmp)=[];
        train.label(tmp)=[];
        Y1=  Y1./( repmat(sqrt(sum(Y1.*Y1)), [m,1]) ); % Normalization.

        % Measurements for test set.
        Y2= phi*test.data;
        energ_of_Y2=sum(Y2.*Y2);
        tmp=find(energ_of_Y2==0);
        Y2(:,tmp)=[];
        test.label(tmp)=[];
        test.data(:, tmp)=[];
        %Y2 = awgn(Y2, Noise, 'measured');
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


        %%%% Save data for the CSEN and OSEN models.
        prox_Y0 = Proj_M * Y0;
        prox_Y1 = Proj_M * Y1;
        tic
        prox_Y2 = Proj_M * Y2;
        toc

        x_dic = zeros(length(Dic.label), maskM, maskN);
        x_train = zeros(length(train.label), maskM, maskN);
        x_test = zeros(length(test.label), maskM, maskN);
        y_dic = zeros(length(Dic.label), maskM, maskN);
        y_train = zeros(length(train.label), maskM, maskN);
        y_test = zeros(length(test.label), maskM, maskN);
        l_dic = Dic.label;
        l_train = train.label;
        l_test = test.label;

        % Segmentation ground-truths.
        for i=1:length(Dic.label)
            x_dic(i,:,:) = reshape(prox_Y0(:, i), maskM, maskN);
            y_dic(i,:,:)=(Dic.label_matrix == Dic.label(i));
        end

        for i=1:length(train.label)
            x_train(i,:,:) = reshape(prox_Y1(:, i), maskM, maskN);
            y_train(i,:,:) = (Dic.label_matrix == train.label(i));
        end

        for i=1:length(test.label)
            x_test(i,:,:) = reshape(prox_Y2(:, i), maskM, maskN);
            y_test(i,:,:)=(Dic.label_matrix==test.label(i));
        end

        save(strcat(outFolder, "\data_dic_", num2str(MR), '_', num2str(k), (".mat")), ...
                    'x_dic', 'x_train', 'x_test', 'y_dic', 'y_train', 'y_test', ...
                    'l_dic', 'l_train', 'l_test', 'Y0', 'Y1', 'Y2', 'Proj_M', '-v6')
        Dic_all(1).label_matrix;
        save(strcat(outFolder, '\dic_label.mat'), 'ans')

    end
    disp('Mean accuracy:')
    disp(mean(per.Rec)) 
end