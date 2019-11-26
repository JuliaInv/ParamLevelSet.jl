
using ParamLevelSet
using jInv.Mesh



println("Testing the rotation itself:")

n = [64,64,64];
Mesh = getRegularMesh([0.0;3.0;0.0;3.0;0.0;3.0],n);
alpha = [1.5;2.5;-2.0;-1.0];
beta = [2.5;2.0;-1.5;2.5];
Xs = [1.5 1.5 1.5 ; 2.0 2.0 2.0; 1.2 2.3 1.5; 2.2 1.5 2.0];
m = wrapTheta(alpha,beta,Xs);

u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0);

theta_phi = deg2rad.([37.0 24.0 ; 27.0 14.0]);
b = [5.0 5.0 7.0 ; -2.3 -5.2 3.5]*(sum(Mesh.h)/3.0);

Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));

N = prod(size(u));
ur1 = zeros(N ,2);
ur2 = copy(ur1);

u = reshape(u,tuple(n...));


@elapsed ur1[:,1] = reshape(rotateAndMove3D(u,theta_phi[1,:],b[1,:]./Mesh.h)[1],N,1);
@elapsed ur1[:,2] = reshape(rotateAndMove3D(u,theta_phi[2,:],b[2,:]./Mesh.h)[1],N,1);

mr2,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 1);
@elapsed ur2[:,1] .= ParamLevelSetModelFunc(Mesh,mr2[:,1];computeJacobian=0,Xc = Xc)[1];
@elapsed ur2[:,2] .= ParamLevelSetModelFunc(Mesh,mr2[:,2];computeJacobian=0,Xc = Xc)[1];

err = norm(ur1-ur2)./(norm(ur1));
if err < 1.5e-1
	println("RBF Rotation test succeeded: ",err);
else
	error("RBF Rotation test not succeeded: ",err);
end


println("Testing the sensitivity of the rotation (simple version - fixed theta/phi and b):");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

m0 = m;
dm = 0.01*randn(length(m));

hh = 1.0;
for k=0:5
	hhh = (0.5^k)*hh;
	mt = rotateAndMoveRBFsimple(m + hhh*dm,Mesh,theta_phi,b;computeJacobian = 0)[1];
	println("norm(mt-mr2): ",norm(mt[:]-mr2[:]),", norm(mt - mr2 - Jrot*dm): ",norm(mt[:] - mr2[:] - Jrot*hhh*dm));
end


println("Testing the sensitivity of the rotation (with varying theta/phi and b):");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
nRBF = div(length(m),5);
nMoves = size(theta_phi,1);
m_wrapped = wrapRBFparamAndRotationsTranslations(m,theta_phi,b);
m_t,theta_phi_t,b_t = splitRBFparamAndRotationsTranslations(m_wrapped,nRBF,nMoves);

if norm(m_t - m) > 1e-5 || norm(theta_phi_t - theta_phi)>1e-5 || norm(b_t - b)>1e-5
	error("wrap/split RBFparamAndRotationsTranslations failed");
end

m0 = wrapRBFparamAndRotationsTranslations(m,theta_phi,b);
dm = 0.01*randn(length(m0));

mr2,Jrot = rotateAndMoveRBF(m,Mesh,theta_phi,b;computeJacobian = 1);

hh = 1.0;
for k=0:10
	hhh = (0.5^k)*hh;
	(mt,theta_phi_t,b_t) = splitRBFparamAndRotationsTranslations(m0+hhh*dm,nRBF,nMoves);
	mt2 = rotateAndMoveRBF(mt,Mesh,theta_phi_t,b_t;computeJacobian = 0)[1];
	println("norm(mt2-mr2): ",norm(mt2[:]-mr2[:]),", norm(mt2 - mr2 - Jrot*dm): ",norm(mt2[:] - mr2[:] - Jrot*hhh*dm));
end








