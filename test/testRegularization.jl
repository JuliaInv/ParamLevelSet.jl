using ParamLevelSet
using jInv.Mesh

println("test for SPD regularizer");

plotting = false;
if plotting
	using jInvVis
	using PyPlot
end

n = [64,64,64];
Mesh = getRegularMesh([0.0;5.0;0.0;5.0;0.0;5.0],n);
midmesh = [(Mesh.domain[2]+Mesh.domain[1])/2 ; (Mesh.domain[4]+Mesh.domain[3])/2; (Mesh.domain[5]+Mesh.domain[6])/2];
lenmesh = [(Mesh.domain[2]-Mesh.domain[1]) ; (Mesh.domain[4]-Mesh.domain[3]); (Mesh.domain[6]-Mesh.domain[5])];
numParamOfRBF = 10;
nRBF = 5
m = zeros(nRBF*numParamOfRBF);
dm = zeros(length(m));

for k=1:nRBF
	offset = (k-1)*numParamOfRBF + 1;
	m[offset] = randn();
	A = randn(3,3); A = 0.3*A'*A + 0.5*Matrix(I,3,3); 
	m[(offset+1):(offset+6)] = A[idx_trilA()];
	m[(offset+7):(offset+9)] = midmesh + 0.05*lenmesh.*randn(size(lenmesh));
	dm[offset] = 0.01*randn(); 
	dm[(offset+1):(offset+6)] = 0.01*randn(6);# 0.01*[randn(); randn(); 0.0; randn(); 0.0; randn()];#
	dm[(offset+7):(offset+9)] = 0.01*randn(3)
end


regfun = (m)-> RBF_SPD_regularization(m,0,nRBF);
hh = 1.0;
R0,dR,d2R = regfun(m);
for k=0:8
	hhh = (0.5^k)*hh;
	Rt,dRt = regfun(m+ hhh*dm);
	println("norm(Rt-R0): ",norm(Rt-R0),", norm(Rt - R0 - dR*dm): ",norm(Rt - R0 - dot(dR,hhh*dm)));
end


println("Testing reg Hessian")
for k=0:8
	hhh = (0.5^k)*hh;
	Rt,dRt = regfun(m+ hhh*dm);
	println("norm(dRt-dR0): ",norm(dRt-dR),", norm(dRt - dR - d2R*dm): ",norm(dRt - dR - d2R*(hhh*dm)));
end


## Below are some checks that we had to do along the way. May be ignored.

# regfun = (A)->(-log(det(A)),-inv(A))
# m = randn(3,3); m = 0.3*m'*m + 0.5*eye(3);
# dm = 0.01*randn(3,3); dm = dm + dm';
# R0,dR = regfun(m);
# for k=0:8
	# hhh = (0.5^k);
	# Rt, = regfun(m+ hhh*dm);
	# println("norm(Rt-R0): ",norm(Rt-R0),", norm(Rt - R0 - dR*dm): ",norm(Rt - R0 - dot(dR[:],hhh*dm[:])));
# end


# regfun = (A)->(-inv(A),kron(inv(A),inv(A)))
# m = randn(3,3); m = 0.3*m'*m + 0.5*eye(3);
# dm = 0.01*randn(3,3); dm = dm + dm';
# R0,dR = regfun(m);
# for k=0:8
	# hhh = (0.5^k);
	# Rt, = regfun(m+ hhh*dm);
	# println("norm(Rt-R0): ",vecnorm(Rt-R0),", vecnorm(Rt - R0 - dR*dm): ",vecnorm(Rt[:] - R0[:] - dR*(hhh*dm[:])));
# end


# itrilA = idx_trilA();
# P = zeros(9,6);
# P[itrilA,1:6] = eye(6);
# P[4,2] = 1.0; P[7,3] = 1.0; P[8,5] = 1.0;
# regfun = (a)->(A = [a[1] a[2] a[3] ; a[2] a[4] a[5] ; a[3] a[5] a[6]]; invA = inv(A); return -invA[itrilA],(kron(invA,invA)*P)[itrilA,:]);
# m = randn(3,3); m = 0.3*m'*m + 0.5*eye(3);
# m = m[itrilA];
# dm = 0.01*randn(6);
# R0,dR = regfun(m);
# for k=0:8
	# hhh = (0.5^k);
	# Rt, = regfun(m+ hhh*dm);
	# println("norm(Rt-R0): ",vecnorm(Rt-R0),", vecnorm(Rt - R0 - dR*dm): ",vecnorm(Rt[:] - R0[:] - dR*(hhh*dm[:])));
# end










