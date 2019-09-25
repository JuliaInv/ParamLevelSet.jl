export rotateAndMoveRBF,rotateAndMoveRBFsimple,wrapRBFparamAndRotationsTranslations,splitRBFparamAndRotationsTranslations, getIpIdxs



function rotateAndMoveRBF(m::Vector{Float64},Mesh::RegularMesh,theta_phi_rad::Array{Float64,2},trans::Array{Float64,2};numParamOfRBF = 5, computeJacobian = 1)
nRBF = div(length(m),numParamOfRBF);
nMoves = size(theta_phi_rad,1);
if nMoves != size(trans,1)
	error("rotateAndMoveRBF: trans vec and theta_phi vec should be of the same length");
end

m_ans = zeros(length(m),nMoves);

for k=1:nMoves
	m_ans[1:numParamOfRBF:end,k] = m[1:numParamOfRBF:end];
end

if numParamOfRBF==5
	for k=1:nMoves
		m_ans[2:numParamOfRBF:end,k] = m[2:numParamOfRBF:end];
	end
else
	P = zeros(9,6);
	P[idx_trilA(),1:6] = getIdentityMat(6);
	P[4,2] = 1.0; P[7,3] = 1.0; P[8,5] = 1.0;
end

mid_domain = [(Mesh.domain[1]+Mesh.domain[2])/2;(Mesh.domain[3]+Mesh.domain[4])/2;(Mesh.domain[5]+Mesh.domain[6])/2]; 
y = zeros(3);
zt = zeros(3);
zp = zeros(3);
o6 = ones(Int64,6);
o3 = ones(Int64,3);


h = 0.0001;

## The jacobian
## Each move has 5 parameters (theta, phi, b1, b2, b3)
## Dims: J: (5*nRBFs + 5*nMoves) -> 5*nRBFs*nMoves
## The rotation affects only the centers x.
## x_new = R*(x - mid) + b + mid

# x_new + dx_new = (R+R_tag_theta*dtheta+R_tag_phi*dphi)*(x+dx-mid) + b + db
# dx_new = R_tag_theta*(x - mid)*dtheta + R_tag_phi*(x - mid)*dphi + R*dx + db 


## for each dip, and basis function, we have:
# dalpha_r		[1  0	 0   0	0 	0]	dalpha  
#  dbeta_r	=	[0	1	 0   0	0 	0]	dbeta
#     dx_r 		[0  0	 R   zt	zp  I]	dx
									  # dtheta
									  # dphi
									  # db
# So, for each dip and basis function, we have 1+1+9+3+3+3 = 20 nz.


## for each dip, and basis function, we have:
# dalpha_r		[1  0	 0   0	0 	0]	dalpha  
#  dAi_r	=	[0	iRiR 0   Tt	Tp 	0]	dAi_r
#  dx_r 		[0  0	 R   zt	zp  I]	dx
									  # dtheta
									  # dphi
									  # db
# So, for each dip and basis function, we have 1+1+9+3+3+3 = 20 nz.



nnzInJacPerMovePerRBF = numParamOfRBF==5 ? 20 : 67;

I = zeros(Int32,nnzInJacPerMovePerRBF*nRBF*nMoves);
J = zeros(Int32,nnzInJacPerMovePerRBF*nRBF*nMoves);
V = zeros(Float64,nnzInJacPerMovePerRBF*nRBF*nMoves);

g_idx = 1;

for k=1:nMoves
	b = vec(trans[k,:]);
	theta_rad = theta_phi_rad[k,1];
	phi_rad = theta_phi_rad[k,2];
	R = getRotate3D(theta_rad,phi_rad);
	invR = inv(R);
	if computeJacobian == 1
		R_tag_phi = (getRotate3D(theta_rad,phi_rad + h) - getRotate3D(theta_rad,phi_rad - h))./(2*h);
		R_tag_theta = (getRotate3D(theta_rad+h,phi_rad) - getRotate3D(theta_rad-h,phi_rad))./(2*h);
		if numParamOfRBF==10
			iRiR = kron(invR',invR');
			iRiR = (iRiR*P)[idx_trilA(),:];
			iR_tag_phi = -invR*R_tag_phi*invR;
			iR_tag_theta = -invR*R_tag_theta*invR;
		end
	end
	for j=1:nRBF
		y[:] = b;
		xj = getX(j,m,numParamOfRBF);
		xj .-= mid_domain;
		BLAS.gemv!('N',1.0,R,xj,1.0,y); # y = y + R*xj;
		y .+= mid_domain;
		m_ans[numParamOfRBF*(j-1).+((numParamOfRBF-2):numParamOfRBF),k] = y;
		if numParamOfRBF==10
			(a1,a2,a3,a4,a5,a6) = getA(j,m);
			A = [a1 a2 a3 ; a2 a4 a5 ; a3 a5 a6];
			Arot = invR'*A*invR;
			m_ans[10*(j-1) .+ (2:7),k] = Arot[idx_trilA()];
		end
		if computeJacobian == 1
			offset = ((k-1)*(numParamOfRBF*nRBF) + (j-1)*numParamOfRBF);
			offsetRBF = numParamOfRBF*(j-1);
			offsetMoves = nRBF*numParamOfRBF + 5*(k-1);
			temp = collect(offset .+ (1:numParamOfRBF));
			BLAS.gemv!('N',1.0,R_tag_theta,xj,0.0,zt);
			BLAS.gemv!('N',1.0,R_tag_phi,xj,0.0,zp);
			y[:] .= 1.0;
			if numParamOfRBF==5
				temp35 = temp[3:5];
				I[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = [temp;temp35;temp35;temp35;temp35;temp35];
				J[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = [offsetRBF.+[1;2;3;3;3;4;4;4;5;5;5]; offsetMoves .+ [1;1;1;2;2;2;3;4;5]];
				V[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = [1.0;1.0;R[:];zt;zp;y];
			else
				Tt = invR'*A*iR_tag_theta; Tt = Tt + Tt'; Tt = Tt[idx_trilA()];
				Tp = invR'*A*iR_tag_phi;   Tp = Tp + Tp'; Tp = Tp[idx_trilA()];
				temp27 = temp[2:7];
				temp810 = temp[8:10];
				I[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = 
					[temp[1];temp27;temp27;temp27;temp27;temp27;temp27;temp810;temp810;temp810;temp27;temp810;temp27;temp810;temp810];
				J[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = 
					[offsetRBF.+[1;2*o6;3*o6;4*o6;5*o6;6*o6;7*o6; 8*o3;9*o3;10*o3]; offsetMoves.+[1*o6;1*o3;2*o6;2*o3;3;4;5]];
				y[:] .= 1.0;
				V[g_idx:(g_idx+nnzInJacPerMovePerRBF-1)] = [1.0;iRiR[:];R[:];Tt;zt;Tp;zp;y];
			end
			g_idx += nnzInJacPerMovePerRBF;
		end
	end
end
Jac = spzeros(0,0);
if computeJacobian == 1
	Jac = sparse(I,J,V,numParamOfRBF*nRBF*nMoves,(numParamOfRBF*nRBF + 5*nMoves));
end
return m_ans,Jac
end

function rotateAndMoveRBFsimple(m::Vector{Float64},Mesh::RegularMesh,theta_phi_rad::Array,trans::Array;computeJacobian = 1,numParamOfRBF::Int64 = 5)
(m_ans,Jac) = rotateAndMoveRBF(m,Mesh,theta_phi_rad,trans;numParamOfRBF = numParamOfRBF,computeJacobian = computeJacobian);
if computeJacobian==1
	Jac = Jac[:,1:length(m)];
end
return (m_ans,Jac);
end


function wrapRBFparamAndRotationsTranslations(m::Vector{Float64},theta_phi_rad::Array,trans::Array)
m_new = zeros(length(m) + length(theta_phi_rad) + length(trans));
m_new[1:length(m)] = m;
m_new[(length(m)+1):end] = vec([theta_phi_rad'; trans']);
return m_new;
end

function splitRBFparamAndRotationsTranslations(m_wraped::Vector{Float64},nRBF::Int64,nMoves::Int64,numParamOfRBF::Int64 = 5)
len_m = (numParamOfRBF*nRBF);
m_RBF = m_wraped[1:len_m];
numParamRotTrans = 5;
theta_phi = [m_wraped[(len_m+1):numParamRotTrans:end] m_wraped[(len_m+2):numParamRotTrans:end]];
translation =  [m_wraped[(len_m+3):numParamRotTrans:end] m_wraped[(len_m+4):numParamRotTrans:end] m_wraped[(len_m+5):numParamRotTrans:end]];
return (m_RBF,theta_phi,translation)
end



function getIpIdxs(I_p, nRBF::Int64,nMoves::Int64,numParamOfRBF::Int64 = 5)
Idxs = zeros(Bool,nRBF*numParamOfRBF + nMoves*5);
for k = I_p
	offset = nRBF*numParamOfRBF + (k-1)*5;
	Idxs[(offset+1):(offset+5)] .= true;
end
Idxs[1:nRBF*numParamOfRBF] = true;
return Idxs;
end





# R = randn(3,3);
# dR = 0.01*randn(3,3);
# u0 = inv(R);
# for k=1:5
	# dR.*=0.5;
	# u = inv(R+dR);
	# ut = u0 - u0*dR*u0;
	# println("norm(ut-u0): ",vecnorm(u-u0),", norm(ut - u0 - J0*dm): ",vecnorm(u - ut));
# end




# function idx_trilA()
	# return [1;2;3;5;6;9];
# end
# # A = randn(3,3); A = A'*A + eye(3);Avec = A[idx_trilA()];
# dA = 0.01*randn(3,3); dA = dA + dA';
# R = randn(3,3);
# dAvec = dA[idx_trilA()];
# P = zeros(9,6);
# P[idx_trilA(),1:6] = eye(6);
# P[4,2] = 1.0; P[7,3] = 1.0; P[8,5] = 1.0;

# U = R'*dA*R; Uvec = U[idx_trilA()];
# RR = (kron(R',R')*P); RR = RR[idx_trilA(),:];
# Vvec = RR*dAvec;
# norm(Vvec - Uvec)





# for k=1:5









