
using ParamLevelSet
using jInv.Mesh
using SparseArrays
using LinearAlgebra
using Test

plotting = false;
if plotting
	using jInvVis
	using PyPlot
end

println("~~~~~~~~~~~~~~~~Testing Mesh Free RBFs.~~~~~~~~~~~~~~~~~~")

n = [32,32,32];
Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
alpha = [1.5;2.5;-2.0;-1.0];
beta = [2.5;2.0;-1.5;2.5];
Xs = [0.5 0.5 0.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
m = wrapTheta(alpha,beta,Xs);



u, = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=0);
if plotting 
	figure()
	u = reshape(u,tuple(Mesh.n...));
	plotModel(u)
end

sigmaH = getDefaultHeaviside();


us, = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH);
if plotting
	figure()
	plotModel(us)
end

bf = 1;
dm = 0.01*randn(length(alpha)*5);


hh = 1.0;
u0,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=1,bf = bf);
J0 = getSparseMatrix(JBuilder);
for k=0:10
	hhh = (0.5^k)*hh;
	ut = MeshFreeParamLevelSetModelFunc(Mesh,m+ hhh*dm;computeJacobian=0,bf = bf)[1];
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - J0*hhh*dm));
end

println("With Heaviside func");

hh = 1.0;

u0,JBuilder = MeshFreeParamLevelSetModelFunc(Mesh,m;computeJacobian=1,sigma=sigmaH,bf = bf);
J0 = getSparseMatrix(JBuilder);

for k=0:10
	hhh = (0.5^k)*hh;
    ut = MeshFreeParamLevelSetModelFunc(Mesh,m + hhh*dm;computeJacobian=0,sigma=sigmaH,bf = bf)[1];
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - J0*hhh*dm));
end





