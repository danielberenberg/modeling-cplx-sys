function dXdt=twospeciespredpreyDiffEq(t,X,a)
% sample call to solve this equation
% [T,Y]=ode45(@predpreyDiffEq,[0:40],[2;1],.75);

X=X(:); % make sure this is a column vector

A=[0.5 0.1; a 0.1]; %interaction matrix (based on the matrix in Flake 12.3)

dXdt = 0*X; % create a vector for derivatives with the right shape
dXdt(1)=X(1).*sum(A(1,:)'.*(1-X(:)));
dXdt(2)=X(2).*sum(A(2,:)'.*(1-X(:)));
