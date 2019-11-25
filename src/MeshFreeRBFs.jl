export MeshFreeParamLevelSetModelFunc
function MeshFreeParamLevelSetModelFunc(Mesh::RegularMesh, m::Vector; computeJacobian = 1, bf::Int64 = 1,
		 sigma::Function = (m,n)->(n[:] .= 1.0),
		 Xc = convert(Array{Float64,2}, getCellCenteredGrid(Mesh)),u::Vector = zeros(size(Xc,1)),dsu::Vector = zeros(size(Xc,1)),
		 Jbuilder::SpMatBuilder{Int64,Float64} = getSpMatBuilder(Int64,Float64,size(Xc,1),length(m),computeJacobian*10*size(Xc,1)),
		 iifunc::Function=identity, numParamOfRBF::Int64 = 5,tree = BallTree(Matrix(Xc')))
	if length(u) != size(Xc,1)
		error("preallocated u is of wrong size");
	end
	u[:] .= 0.0;
	reset!(Jbuilder);
	nRBFs = div(length(m),numParamOfRBF);
	## in run 1 we calculate u. in run 2 we calculate J after we know the derivative of the heaviside. 
	beta_arr = Array{Float64}(undef,nRBFs);
		
	if numParamOfRBF==5
		for k=1:nRBFs
			MeshFreeComputeFunc!(m,k,u,Xc,tree);
		end
	else
		for k=1:nRBFs
			(a1,a2,a3,a4,a5,a6) = getA(k,m);
			A = [a1 a2 a3 ; a2 a4 a5 ; a3 a5 a6];
			beta = sqrt(minimum(eigvals(A)));
			MeshFreeComputeFunc_ellipsoids!(m,k,u,Xc,tree,beta);
			beta_arr[k] = beta;
		end
	end
	sigma(u,dsu);
	if computeJacobian == 1
		if numParamOfRBF==5
			for k=1:nRBFs
				MeshFreeUpdateJacobian!(k,m,dsu,Jbuilder,u,iifunc,Xc,tree);
			end
		else
			for k=1:nRBFs
				MeshFreeUpdateJacobian_ellipsoids!(k,m,dsu,Jbuilder,u,iifunc,Xc,tree,beta_arr[k])
			end
		end	
	end
	return u,Jbuilder;
end



function MeshFreeComputeFunc!(m::Vector{Float64},k::Int64,u::Vector{Float64},Xc,tree)
	(x1,x2,x3) = getXt(k,m);
	beta 	= getBeta(k,m);
	betaSq 	= beta^2;
	thres 	= 0.81/betaSq;# 0.77~0.875^2
	alpha 	= getAlpha(k,m);
	idxs = inrange(tree, [x1,x2,x3], sqrt(thres));
	for ii = idxs
		@inbounds y = Xc[ii,1]-x1; y*=y;
		nX = y;
		@inbounds y = Xc[ii,2]-x2; y*=y;
		nX+=y;
		@inbounds y = Xc[ii,3]-x3; y*=y;
		nX+=y;
		if (nX <= thres) 
			nX*=betaSq;
			argii = radiust(nX);
			argii = psi1(argii);
			argii *= alpha;
			@inbounds u[ii] += argii;
		else
			println("WE SHOULD NOT GET HERE!!!!!");
		end
	end
end


function MeshFreeUpdateJacobian!(k::Int64,m::Vector{Float64},dsu::Vector{Float64},Jbuilder::SpMatBuilder,u::Vector{Float64},iifunc::Function,Xc,tree)
	numParamOfRBF = 5;
	(x1,x2,x3) = getXt(k,m);
	alpha = getAlpha(k,m);
	beta = getBeta(k,m);
	betaSQ = beta*beta;
	invBeta = (1.0/beta);
	thres = 0.81/betaSQ;

	alphaBetaSq = alpha*betaSQ;
	offset = convert(Int64,(k-1)*numParamOfRBF + 1);
	md = 1e-3*maximum(abs.(dsu));

	idxs = inrange(tree, [x1,x2,x3], sqrt(thres));

	for ii = idxs
		temp = dsu[ii];
		if temp >= md
			@inbounds y1 = x1 - Xc[ii,1]; nX =y1*y1;
			@inbounds y2 = x2 - Xc[ii,2]; nX+=y2*y2;
			@inbounds y3 = x3 - Xc[ii,3]; nX+=y3*y3;
			if (nX <= thres) # 0.77~0.875^2
				radii = radiust(nX*betaSQ);
				psi,dpsi = dpsi1_t(radii);
				psi*=temp;
				temp*= alphaBetaSq;
				temp/= radii;
				temp*=dpsi;
				nX *= temp;
				nX *= invBeta;
				y1*=temp; y2*=temp; y3*=temp;
				ii_t = iifunc(ii);
				setNext!(Jbuilder,ii_t,offset,psi);
				setNext!(Jbuilder,ii_t,offset+1,nX);
				setNext!(Jbuilder,ii_t,offset+2,y1);
				setNext!(Jbuilder,ii_t,offset+3,y2);
				setNext!(Jbuilder,ii_t,offset+4,y3);		
			end
		end
	end
end

function MeshFreeComputeFunc_ellipsoids!(m::Vector{Float64},k::Int64,u::Vector{Float64},Xc::Array,tree,beta::Float64)
numParamOfRBF = 10;
(x1,x2,x3) = getXt(k,m,numParamOfRBF);
(a1,a2,a3,a4,a5,a6) = getA(k,m);
alpha = getAlpha(k,m,numParamOfRBF);
idxs = inrange(tree, [x1,x2,x3], 0.9/beta);
for ii = idxs
	nX = 0.0;
	@inbounds y1 = Xc[ii,1]-x1;
	@inbounds y2 = Xc[ii,2]-x2;
	@inbounds y3 = Xc[ii,3]-x3;
	Ay1,Ay2,Ay3 = mulA(a1,a2,a3,a4,a5,a6,y1,y2,y3);
	y1*=Ay1;y2*=Ay2;y3*=Ay3; 
	nX+=y1;nX+=y2;nX+=y3;
	if (nX <= 0.81) # 0.77~0.875^2
		argii = radiust(nX);
		argii = psi1(argii);
		argii *= alpha;
		@inbounds u[ii] += argii;
	end
end
end


function MeshFreeUpdateJacobian_ellipsoids!(k::Int64,m::Vector{Float64},dsu::Vector{Float64},Jbuilder::SpMatBuilder,u::Vector{Float64},iifunc::Function,Xc,tree,beta::Float64)
numParamOfRBF = 10;
(x1,x2,x3) = getXt(k,m,numParamOfRBF);
(a1,a2,a3,a4,a5,a6) = getA(k,m);
alpha = getAlpha(k,m,numParamOfRBF);
offset = convert(Int64,(k-1)*numParamOfRBF + 1);
md = 1e-3*maximum(abs.(dsu));
temp = 0.0;
idxs = inrange(tree, [x1,x2,x3], 0.9/beta);
for ii = idxs
	temp = dsu[ii];
	if temp >= md
		nX = 0.0;
		@inbounds z1 = x1 - Xc[ii,1]; 
		@inbounds z2 = x2 - Xc[ii,2]; 
		@inbounds z3 = x3 - Xc[ii,3];
		(Az1,Az2,Az3) = mulA(a1,a2,a3,a4,a5,a6,z1,z2,z3);
		nX = z1*Az1 + z2*Az2 + z3*Az3;
		if (nX <= 0.81) # 0.77~0.875^2
			radii = radiust(nX);
			psi,dpsi = dpsi1_t(radii);
			psi*=temp;
			ii_t = iifunc(ii);
			setNext!(Jbuilder,ii_t,offset,psi);
			temp *= alpha;
			temp /= radii;
			temp *= dpsi;
			psi = z1; psi*=temp;
			setNext!(Jbuilder,ii_t,offset+1,0.5*z1*psi);
			setNext!(Jbuilder,ii_t,offset+2,z2*psi);
			setNext!(Jbuilder,ii_t,offset+3,z3*psi);
			psi = z2; psi*=temp;z2*=0.5;z2*=psi;
			setNext!(Jbuilder,ii_t,offset+4,z2);
			setNext!(Jbuilder,ii_t,offset+5,z3*psi);
			setNext!(Jbuilder,ii_t,offset+6,0.5*z3*z3*temp);
			Az1*=temp;Az2 *= temp; Az3 *= temp; 
			setNext!(Jbuilder,ii_t,offset+7,Az1);
			setNext!(Jbuilder,ii_t,offset+8,Az2);
			setNext!(Jbuilder,ii_t,offset+9,Az3);			
		end
	end
end
end



