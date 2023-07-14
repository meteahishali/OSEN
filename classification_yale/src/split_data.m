function [ Dic, train, test ] = split_data( data,labels,param )

class_n=length(unique(labels)); %number of class (people)
N1=param.dictionary_size; 
N2=param.train_size;
N3=param.test_size;

%MS=param.matrixsize; %[19,2]
gds=[1; unique(cumprod(perms(factor(class_n)),2))];
m1=gds(end-1);
m2=class_n/m1;
gds=[1; unique(cumprod(perms(factor(N1)),2))];
m22=gds(floor((length(gds)-2)/2)+1);
m11=N1/m22;

%%initilization
D=zeros(size(data,1),N1*class_n);
train.data = zeros(size(data,1),N2*class_n);
test.data =  zeros(size(data,1),size(data,2)-(N2+N1)*class_n);
train.label=zeros(N2*class_n,1);
test.label=zeros(size(data,2)-(N2+N1)*class_n,1);

%%
temp= [];
A=[];
t=1;
for k=1:m2
    for l=1:m1
        temp=[temp,ones(m11,m22)*t];
        t=t+1;
    end
    A=[A;temp];
    temp=[];    
end

for i=1:class_n
    in=find(A(:)==i);
    in2=find(labels==i);
    %rng(42);
    %rng(92);
    %rng(192);
    %rng(122);
    %rng(236);

    in2=in2(randperm(length(in2)));
    for k=1:length(in)
        D(:,in(k))=data(:,in2(k));
%        figure(99),imshow(reshape(D(:,k-1),32,32),[])
    end
    for l=1:N2
        train.data(:,N2*(i-1)+l)=data(:,in2(k+l));
        train.label(N2*(i-1)+l)=i;
    end
    for t=1:(length(in2)-N2-N1)
        test.data(:,N3*(i-1)+t)=data(:,in2(k+l+t));
        test.label(N3*(i-1)+t)=i;
    end
Dic.dictionary=D;
Dic.label=A(:);
Dic.label_matrix=A;
end

