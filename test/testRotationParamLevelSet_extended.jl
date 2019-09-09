
using ParamLevelSet
using jInv.Mesh
using LinearAlgebra



println("Testing the rotation itself (extended RBFs):")

n = [64,64,64];
Mesh = getRegularMesh([0.0;5.0;0.0;5.0;0.0;5.0],n);
midmesh = [(Mesh.domain[2]+Mesh.domain[1])/2 ; (Mesh.domain[4]+Mesh.domain[3])/2; (Mesh.domain[5]+Mesh.domain[6])/2];
lenmesh = [(Mesh.domain[2]-Mesh.domain[1]) ; (Mesh.domain[4]-Mesh.domain[3]); (Mesh.domain[6]-Mesh.domain[5])];
numParamOfRBF = 10;
nRBF = 5
m = zeros(nRBF*numParamOfRBF);
for k=1:nRBF
	offset = (k-1)*numParamOfRBF + 1;
	m[offset] = randn();
	#A = randn(3,3); A = 0.3*A'*A + 0.5*eye(3); 
	A = diagm(rand(3).+1.0);
	# L = chol(A)';
	# m[(offset+1):(offset+6)] = L[L.!=0.0];
	m[(offset+1):(offset+6)] = A[idx_trilA()];
	m[(offset+7):(offset+9)] = midmesh .+ 0.05*lenmesh.*randn(size(lenmesh));
end
u, = ParamLevelSetModelFunc(Mesh,m;computeJacobian=0,numParamOfRBF=numParamOfRBF);

theta_phi = deg2rad.([37.0 24.0 ; 27.0 14.0]);
b = [1.0 5.0 7.0 ; -2.3 5.2 3.5]*sum((Mesh.h))/3.0;


Xc = convert(Array{Float32,2},getCellCenteredGrid(Mesh));

N = prod(size(u));
ur1 = zeros(N ,2);
ur2 = copy(ur1);

u = reshape(u,tuple(n...));

@elapsed ur1[:,1] = reshape(rotateAndMove3D(u,theta_phi[1,:],b[1,:]./Mesh.h)[1],N,1);
@elapsed ur1[:,2] = reshape(rotateAndMove3D(u,theta_phi[2,:],b[2,:]./Mesh.h)[1],N,1);;

mr2,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 0,numParamOfRBF=numParamOfRBF);
@elapsed ur2[:,1] = ParamLevelSetModelFunc(Mesh,mr2[:,1];computeJacobian=0,Xc = Xc,numParamOfRBF=numParamOfRBF)[1];
@elapsed ur2[:,2] = ParamLevelSetModelFunc(Mesh,mr2[:,2];computeJacobian=0,Xc = Xc,numParamOfRBF=numParamOfRBF)[1];



err = norm(ur1-ur2)./(2*norm(u));
if err < 2e-1
	println("RBF Rotation test succeeded: ",err);
else
	error("RBF Rotation test not succeeded: ",err);
end


println("Testing the sensitivity of the rotation (simple version - fixed theta/phi and b):");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
mr2,Jrot = rotateAndMoveRBFsimple(m,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);
m0 = m;
dm = 0.01*randn(length(m));

# for k=1:nRBF
	# offset = (k-1)*numParamOfRBF + 1;
	# dm[offset] = 0.01*randn();
	# dm[(offset+1):(offset+6)] = 0.01*randn(6);
	# dm[(offset+7):(offset+9)] = 0.01*randn(3);
# end

hh = 1.0;
for k=0:5
	hhh = (0.5^k)*hh;
	mt = rotateAndMoveRBFsimple(m + hhh*dm,Mesh,theta_phi,b;computeJacobian = 0,numParamOfRBF=numParamOfRBF)[1];
	mtt= mr2 + reshape(Jrot*hhh*dm,size(mr2));
	println("norm(mt-mr2): ",norm(mt[:]-mr2[:]),", norm(mt - mr2 - Jrot*dm): ",norm(mt[:] - mr2[:] - Jrot*hhh*dm));
end


println("Testing the sensitivity of the rotation (with varying theta/phi and b):");
println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
nRBF = div(length(m),numParamOfRBF);
nMoves = size(theta_phi,1);
m_wrapped = wrapRBFparamAndRotationsTranslations(m,theta_phi,b);
m_t,theta_phi_t,b_t = splitRBFparamAndRotationsTranslations(m_wrapped,nRBF,nMoves,numParamOfRBF);

if norm(m_t - m) > 1e-5 || norm(theta_phi_t - theta_phi)>1e-5 || norm(b_t - b)>1e-5
	error("wrap/split RBFparamAndRotationsTranslations failed");
end

m0 = wrapRBFparamAndRotationsTranslations(m,theta_phi,b);
dm = 0.01*randn(length(m0));

dm,dtheta_phi,db = splitRBFparamAndRotationsTranslations(m_wrapped,nRBF,nMoves,numParamOfRBF);
dm = wrapRBFparamAndRotationsTranslations(dm,dtheta_phi,db);



mr2,Jrot = rotateAndMoveRBF(m,Mesh,theta_phi,b;computeJacobian = 1,numParamOfRBF=numParamOfRBF);

hh = 1.0;
for k=0:10
	hhh = (0.5^k)*hh;
	(mt,theta_phi_t,b_t) = splitRBFparamAndRotationsTranslations(m0+hhh*dm,nRBF,nMoves,numParamOfRBF);
	mt2 = rotateAndMoveRBF(mt,Mesh,theta_phi_t,b_t;computeJacobian = 0,numParamOfRBF=numParamOfRBF)[1];
	println("norm(mt2-mr2): ",norm(mt2[:]-mr2[:]),", norm(mt2 - mr2 - Jrot*dm): ",norm(mt2[:] - mr2[:] - Jrot*hhh*dm));
end








