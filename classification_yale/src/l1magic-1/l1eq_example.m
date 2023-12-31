% l1eq_example.m
%
% Test out l1eq code (l1 minimization with equality constraints).
%
% Written by: Justin Romberg, Caltech
% Email: jrom@acm.caltech.edu
% Created: October 2005
%

% put key subdirectories in path if not already there
path(path, './Optimization');
path(path, './Data');

% To reproduce the example in the documentation, uncomment the 
% two lines below
%load RandomStates
%rand('state', rand_state);
%randn('state', randn_state);

% signal length
N = 256*256;
% number of spikes in the signal
T = 20;
% number of observations to make
K = 2000;

% random +/- 1 signal
x = zeros(N,1);
q = randperm(N);
x(q(1:T)) = sign(randn(T,1));

% measurement matrix
% disp('Creating measurment matrix...');
% A = randn(K,N);
% A = orth(A')';
% disp('Done.');
tic 
Afun = @(z) m_matrix(z,K,N);
y=Afun(x);
toc



Atfun = @(z) m_matrix_tra(z,K,N);

% observations
% y = A*x;

% initial guess = min energy
%x0 = A'*y;
tic
x0 = Atfun(y);
toc
d=1;
% solve the LP
% tic
% xp = l1eq_pd(x0, A, [], y, 1e-3);
% toc

% large scale

tic
xp = l1eq_pd(x0, Afun, Atfun, y, 1e-3, 30, 1e-8, 200);
toc

figure, plot(1:N,x,'b+',1:N,xp,'ro'), legend('original','Proposed Solution of cover work')
title(norm((x-xp),2))


