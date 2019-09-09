export RBF_SPD_regularization
function RBF_SPD_regularization(m::Array{Float64,1},mref,nRBF)

R = 0.0;
dR = zeros(size(m));
II = ones(Int64,nRBF*36);
JJ = ones(Int64,nRBF*36);
VV = zeros(Float64,nRBF*36);
I3 = Matrix(1.0I,3,3);
o6 = ones(Int64,6);
itrilA = idx_trilA();

ineqj = [2;3;5];

P = zeros(9,6);
P[itrilA,1:6] = Matrix(1.0I,6,6);
P[4,2] = 1.0; P[7,3] = 1.0; P[8,5] = 1.0;

for k=1:nRBF
	(a1,a2,a3,a4,a5,a6) = getA(k,m);
	A     = [a1 a2 a3 ; a2 a4 a5 ; a3 a5 a6];
	chol = 0;
	try
		chol = cholesky(A);
	catch
		R = Inf;
		break;
	end
	R      -= 2.0*log(det(chol.L));
	invA    = chol\I3;
	# invA = inv(A);
	# R -= log(det(A));
	iAiA 	= (kron(invA,invA)*P)[itrilA,:];
	offset  = 10*(k-1);
	AIDXs = (2+offset):(7+offset);
	invA = -invA[itrilA];
	invA[ineqj].*=2.0;
	iAiA[ineqj,:].*=2.0;
	dR[AIDXs] = invA;
	II[((k-1)*36+1) : (k*36)] = [o6*AIDXs[1];o6*AIDXs[2];o6*AIDXs[3];o6*AIDXs[4];o6*AIDXs[5];o6*AIDXs[6]];
	JJ[((k-1)*36+1) : (k*36)] = [AIDXs;AIDXs;AIDXs;AIDXs;AIDXs;AIDXs];
	VV[((k-1)*36+1) : (k*36)] = iAiA[:];
end
d2R = sparse(II,JJ,VV,length(m),length(m));
return R,dR,d2R;
end

