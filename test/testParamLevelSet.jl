
using ParamLevelSet
using jInv.Mesh
using SparseArrays
using LinearAlgebra

plotting = false;
if plotting
	using jInvVis
	using PyPlot
end

n = [64,64,64];
Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
alpha = [1.5;2.5;-2.0;-1.0];
beta = [2.5;2.0;-1.5;2.5];
Xs = [0.5 0.5 0.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
m = wrapTheta(alpha,beta,Xs);



u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0);
if plotting 
	figure()
	u = reshape(u,tuple(Mesh.n...));
	plotModel(u)
end

sigmaH = getDefaultHeavySide();


us, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,sigma = sigmaH);
if plotting
	figure()
	plotModel(us)
end

bf = 1;
dm = 0.01*randn(length(alpha)*5);

# dm[2:5:end] = 0.0;

# dm[1:5:end] = 0.0;
# dm[3:5:end] = 0.0;
# dm[4:5:end] = 0.0;
# dm[5:5:end] = 0.0;


hh = 1.0;
u0,II,JJ,VV = ParamLevelSetModelFunc(Mesh,m;computeJacobian=1,bf = bf);
J0 = sparse(II,JJ,VV,prod(Mesh.n),length(m))
for k=0:10
	hhh = (0.5^k)*hh;
	ut = ParamLevelSetModelFunc(Mesh,m+ hhh*dm;computeJacobian=0,bf = bf)[1];
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - J0*hhh*dm));
end

println("With Heavy Side func");

hh = 1.0;

u0,II,JJ,VV = ParamLevelSetModelFunc(Mesh,m;computeJacobian=1,sigma=sigmaH,bf = bf);
J0 = sparse(II,JJ,VV,prod(Mesh.n),length(m));

for k=0:10
	hhh = (0.5^k)*hh;
	
    ut = ParamLevelSetModelFunc(Mesh,m + hhh*dm;computeJacobian=0,sigma=sigmaH,bf = bf)[1];
	
	println("norm(ut-u0): ",norm(ut[:]-u0[:]),", norm(ut - u0 - J0*dm): ",norm(ut[:] - u0[:] - J0*hhh*dm));
end




println("Testing Heavyside")
u = randn(10);
(us1,ds1) = heavySide(u,0.0);
ds2 = zeros(size(u));
heavySide!(u,ds2,0.0);
println(norm(u - us1));
println(norm(diag(ds1) - ds2));







