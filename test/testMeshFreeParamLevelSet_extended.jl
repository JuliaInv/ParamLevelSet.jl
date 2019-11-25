using ParamLevelSet
using jInv.Mesh
using LinearAlgebra 
using SparseArrays
using Test

plotting = false;
if plotting
	using jInvVis
	using PyPlot
end

println("~~~~~~~~~~~~~~~~Testing Mesh Free extended RBFs.~~~~~~~~~~~~~~~~~~")

n = [32,32,32];

Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
midmesh = [(Mesh.domain[2]+Mesh.domain[1]/2) ; (Mesh.domain[4]+Mesh.domain[3]/2); (Mesh.domain[5]+Mesh.domain[6]/2)];
lenmesh = [(Mesh.domain[2]-Mesh.domain[1]) ; (Mesh.domain[4]-Mesh.domain[3]); (Mesh.domain[6]-Mesh.domain[5])];
numParamOfRBF = 10;
nRBF = 5
m5 = zeros(nRBF*5);
m10 = zeros(nRBF*numParamOfRBF);

for k=1:nRBF
	offset10 = (k-1)*numParamOfRBF + 1;
	m10[offset10] = randn();
	beta = max(abs(randn()),0.5);
	# println(beta)
	# m10[offset10+1] = beta; m10[offset10+4] = beta; m10[offset10+6] = beta; ## for the L version.
	m10[offset10+1] = beta^2; m10[offset10+4] = beta^2; m10[offset10+6] = beta^2; ## for the A version.
	m10[(offset10+7):(offset10+9)] = midmesh + 0.1*lenmesh.*randn(size(lenmesh));
	offset5 = (k-1)*5 + 1;
	m5[offset5] = m10[offset10];
	m5[offset5+1] = beta;
	m5[(offset5+2):(offset5+4)] = m10[(offset10+7):(offset10+9)];
end

u5, = MeshFreeParamLevelSetModelFunc(Mesh,m5;computeJacobian=0,numParamOfRBF = 5);
u10, = MeshFreeParamLevelSetModelFunc(Mesh,m10;computeJacobian=0,numParamOfRBF = 10);
if norm(u5-u10) > 1e-5
	error(string("This should be zero: ",norm(u5-u10)));
end

n = [32,32,32];
Mesh = getRegularMesh([0.0;5.0;0.0;5.0;0.0;5.0],n);
midmesh = [(Mesh.domain[2]+Mesh.domain[1])/2 ; (Mesh.domain[4]+Mesh.domain[3])/2; (Mesh.domain[5]+Mesh.domain[6])/2];
lenmesh = [(Mesh.domain[2]-Mesh.domain[1]) ; (Mesh.domain[4]-Mesh.domain[3]); (Mesh.domain[6]-Mesh.domain[5])];
numParamOfRBF = 10;
nRBF = 5
m = zeros(nRBF*numParamOfRBF);

for k=1:nRBF
	offset = (k-1)*numParamOfRBF + 1;
	m[offset] = randn();
	A = randn(3,3); A = 0.5*A'*A + 0.5*Matrix(1.0I,3,3); #A = eye(3) + 1e-5;
	m[(offset+1):(offset+6)] = A[tril(A).!=0.0];
	m[(offset+7):(offset+9)] = midmesh + 0.05*lenmesh.*randn(size(lenmesh));
end
u, = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=0,numParamOfRBF = 10);
ufull, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,numParamOfRBF = 10);
@test norm(u-ufull) < 1e-8


if plotting 
	figure()
	u = reshape(u,tuple(Mesh.n...));
	plotModel(u)
end

sigmaH = getDefaultHeaviside();
us, = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH,numParamOfRBF = 10);
if plotting
	figure()
	plotModel(us)
end

bf = 1;
dm = 0.01*zeros(length(m));

for k=1:nRBF
	offset = (k-1)*numParamOfRBF + 1;
	dm[offset] = 0.01*randn(); 
	dm[(offset+1):(offset+6)] = 0.01*randn(6);# 0.01*[randn(); randn(); 0.0; randn(); 0.0; randn()];#
	dm[(offset+7):(offset+9)] = 0.01*randn(3)
end

hh = 1.0;

u0,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=1,bf = bf,numParamOfRBF = 10);
J0 = getSparseMatrix(JBuilder);
ufull,JBuilder = ParamLevelSetModelFunc(Mesh,m;computeJacobian=1,bf = bf,numParamOfRBF = 10);
Jfull = getSparseMatrix(JBuilder);

@test norm(Jfull - J0,1) < 1e-8

for k=0:8
	hhh = (0.5^k)*hh;
	ut = MeshFreeParamLevelSetModelFunc(Mesh,m+ hhh*dm;computeJacobian=0,bf = bf,numParamOfRBF = 10)[1];
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - J0*hhh*dm));
end

println("With Heaviside func");

hh = 1.0;
u0,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=1,sigma=sigmaH,bf = bf,numParamOfRBF = 10);
J0 = getSparseMatrix(JBuilder);
for k=0:8
	hhh = (0.5^k)*hh;
    ut = MeshFreeParamLevelSetModelFunc(Mesh,m + hhh*dm;computeJacobian=0,sigma=sigmaH,bf = bf,numParamOfRBF = 10)[1];
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - hhh*(J0*dm)));
end







