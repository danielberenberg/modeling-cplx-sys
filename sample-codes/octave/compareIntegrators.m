function compareIntegrators(h,maxT,X0,a)
% This code was developed and tested in GNU Octave, not Matlab.
% Calls to special function (e.g. ode45) might be slightly different.

% Note for octave: pkg load odepkg
% Sample call: compareIntegrators(0.1, 50, [0.2; 0.4], 0.25)

% Arguments: h is the integration step,
%            maxT is the target final time
%            X0 is a vector of initial conditions for all variables
%            a is a parameter in the equations

X0=X0(:); % make sure this is a column vector
T=[0:h:maxT]; % all time steps

[T,Y]=ode45(@twospeciespredpreyDiffEq,T,X0,a);
%[T,Ye]=euler(@predpreyDiffEq,T,X0,[],a);
%[T,Yh]=heun(@predpreyDiffEq,T,X0,[],a);

plot(T,Y)
