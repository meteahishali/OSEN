function [x_BB_mono] = solveGPSRBB(y,A)

% regularization parameter
tau = 0.005*max(abs(A'*y));

first_tau_factor = 0.8*(max(abs(A'*y))/tau);
steps = 5;

debias = 0;

stopCri=3;
tolA=1.e-5;

%disp('Starting GPSR BB monotonic')
[x_BB_mono,x_debias_BB_mono,obj_BB_mono,...
    times_BB_mono,debias_start_BB_mono,mse_BB_mono]= ...
         GPSR_BB(y,A,tau,...
         'Debias',debias,...
         'Monotone',1,...
         'Initialization',0,...
         'MaxiterA',10000,...
         'StopCriterion',stopCri,...
       	 'ToleranceA',tolA,...
         'Verbose',0);
     
end
