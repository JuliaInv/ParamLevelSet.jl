export getAlpha, getBeta, getX,ParamLevelSetModelFunc
function ParamLevelSetModelFunc(Mesh::RegularMesh,m::Vector; computeJacobian = 1, bf::Int64 = 1,
		 sigma::Function = (m,n)->(n[:] .= 1.0),
		 Xc = convert(Array{Float32,2}, getCellCenteredGrid(Mesh)),u::Vector = zeros(prod(Mesh.n)),dsu::Vector = zeros(prod(Mesh.n)),
		 Jbuilder::SpMatBuilder{Int64,Float64} = getSpMatBuilder(Int64,Float64,prod(Mesh.n),length(m),computeJacobian*10*prod(Mesh.n)),
		 iifunc::Function=identity, numParamOfRBF::Int64 = 5)
	n = Mesh.n;
	if length(u) != prod(n)
		error("preallocated u is of wrong size");
	end
	u[:] .= 0.0;
	reset!(Jbuilder);
	h = Mesh.h;
	nRBFs = div(length(m),numParamOfRBF);
	xmin = Mesh.domain[[1;3;5]];
	beta_arr = Array{Float64}(undef,nRBFs);
		
	
	## in run 1 we calculate u. in run 2 we calculate J after we know the derivative of the heaviside. 
	for k=1:nRBFs
		if numParamOfRBF==5
			computeFunc!(m,k,Xc,h,n,u,xmin);
		else
			# putting this inside the function takes a huge amount of time... go figure..
			(a1,a2,a3,a4,a5,a6) = getA(k,m);
			A = [a1 a2 a3 ; a2 a4 a5 ; a3 a5 a6];
			beta = sqrt(minimum(eigvals(A)));
			computeFunc_ellipsoids!(m,k,Xc,h,n,u,xmin,beta);
			beta_arr[k] = beta;
		end
	end
	sigma(u,dsu);
	nnzJ = 0
	if computeJacobian == 1
		if numParamOfRBF==5
			for k=1:nRBFs
				updateJacobian!(k,m,dsu,h,n,Jbuilder,u,iifunc,Xc,xmin);
			end
		else
			for k=1:nRBFs
				updateJacobian_ellipsoids!(k,m,dsu,h,n,Jbuilder,u,iifunc,Xc,xmin,beta_arr[k])
			end
		end	
	end
	return u,Jbuilder;
end



function computeFunc!(m::Vector{Float64},k::Int64,Xc,h::Vector{Float64},n::Vector{Int64},
         u::Vector{Float64},xmin::Vector{Float64})
(x1,x2,x3) = getXt(k,m);
i = round.(Int64,(getX(k,m) - xmin)./h .+ 0.5);
# ## here we take a reasonable box around the bf. 
# ## The constant depends on the type of RBF that we take.
beta = getBeta(k,m);
betaSq = beta^2;
thres = 0.81/betaSq;
boxL = ceil.(Int64,0.9./(h*abs(beta)));
imax = min.(i + boxL,n);
imin = max.(i - boxL,[1;1;1]);
alpha = getAlpha(k,m);

@inbounds for l = imin[3]:imax[3]
	@inbounds iishift3 = (l-1)*n[1]*n[2];
	@inbounds for j = imin[2]:imax[2]
		@inbounds iishift2 = iishift3 + (j-1)*n[1];
		@inbounds for q = imin[1]:imax[1]
			ii = iishift2 + q;
			@inbounds y = Xc[ii,1]-x1; y*=y;
			nX = y;
			@inbounds y = Xc[ii,2]-x2; y*=y;
			nX+=y;
			@inbounds y = Xc[ii,3]-x3; y*=y;
			nX+=y;
			if (nX <= thres) # 0.77~0.875^2
				nX*=betaSq;
				argii = radiust(nX);
				argii = psi1(argii);
				argii *= alpha;
				@inbounds u[ii] += argii;
			end
		end
	end
end
end


function updateJacobian!(k::Int64,m::Vector{Float64},dsu::Vector{Float64},h::Vector{Float64},n::Vector{Int64},Jbuilder::SpMatBuilder,u::Vector{Float64},iifunc::Function,Xc,xmin::Vector{Float64})
(x1,x2,x3) = getXt(k,m);
alpha = getAlpha(k,m);
beta = getBeta(k,m);
betaSQ = beta*beta;
invBeta = (1.0/beta);
thres = 0.81/betaSQ;


alphaBetaSq = alpha*betaSQ;
offset = convert(Int64,(k-1)*5 + 1);
md = 1e-3*maximum(abs.(dsu));
boxL = ceil.(Int64,0.9./(h*abs(beta)));
i = round.(Int64,([x1;x2;x3] - xmin)./h .+ 0.5);
imax = min.(i + boxL,n);
imin = max.(i - boxL,[1;1;1]);

@inbounds for l = imin[3]:imax[3]
	@inbounds iishift3 = (l-1)*n[1]*n[2];
	@inbounds for j = imin[2]:imax[2]
		@inbounds iishift2 = iishift3 + (j-1)*n[1];
		@inbounds for q = imin[1]:imax[1]
			ii = iishift2 + q;
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
					ii = iifunc(ii);
					setNext!(Jbuilder,ii,offset,psi);
					setNext!(Jbuilder,ii,offset+1,nX);
					setNext!(Jbuilder,ii,offset+2,y1);
					setNext!(Jbuilder,ii,offset+3,y2);
					setNext!(Jbuilder,ii,offset+4,y3);		
				end
			end
		end
	end
end
end 


function getAlpha(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
	return m[numParamOfRBF*(k-1) + 1];
end

function getBeta(k::Int64,m::Vector{Float64})
	return m[5*(k-1) + 2];
end

function getX(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
	return m[numParamOfRBF*(k-1).+((numParamOfRBF-2):numParamOfRBF)];
end

function getXt(k::Int64,m::Vector{Float64},numParamOfRBF::Int64 = 5)
	return (m[numParamOfRBF*(k-1)+numParamOfRBF-2],m[numParamOfRBF*(k-1)+numParamOfRBF-1],m[numParamOfRBF*(k-1)+numParamOfRBF]);
end

function setX(k::Int64,m::Vector{Float64},x::Array)
	m[5*(k-1)+(3:5)] = x;
end

export psi
function psi(r::Union{Array{Float64,1},Array{Float32,1}},bf = 1)
	rpp = max.(1.0-r,0.0);
	if bf==1
		psi = (rpp.^4).*(4.0*r + 1.0);
	elseif bf == 2
		psi = (rpp.^6).*((11.6666666)*r.^2 + 6.0*r + 1.0); ### this is the RBF divided by 3. No idea what went wrong there...
	elseif bf == 3
		psi = (rpp.^8).*(32.0*(r.^3) + 25.0.*(r.^2) + 8.0.*r + 1.0);
	end
	return psi;
end

# function dpsi(r::Union{Array{Float64,1},Array{Float32,1}},bf = 1)
	# rpp = max.(1.0-r,0.0);
	# if bf==1
		# dpsi = -4.0*(rpp.^3).*(4.0*r + 1.0) + (rpp.^4).*4.0;
	# elseif bf == 2
		# dpsi = -6.0*(rpp.^5).*((11.6666666)*r.^2 + 6.0*r + 1.0) + (rpp.^6).*((2.0*11.6666666)*r + 6.0); ### this is the RBF divided by 3. No idea what went wrong there...
	# elseif bf == 3
		# dpsi = -8.0*(rpp.^7).*(32.0*(r.^3) + 25.*(r.^2) + 8.0.*r + 1.0) + (rpp.^8).*(96.0*(r.^2) + 50.*r + 8.0);
	# end
	# return dpsi;
# end

function psi1(r::Float64)
	psi = 1.0;
	psi -= r;
	r *= 4.0;
	r += 1.0;
	psi*=psi;
	psi*=psi;
	psi*=r;
	return psi;
end

function dpsi1(r::Float64)
	dpsi =1.0;
	dpsi -= r;
	r *= -4.0;
	r -= 1.0;
	r += dpsi;
	dpsi *= (dpsi*dpsi);
	dpsi *= 4.0;
	dpsi *= r;
	return dpsi;
end

function dpsi1_t(r::Float64)
	psi = 1.0;
	psi -= r;
	r *= 4.0;
	r += 1.0;
	dpsi = psi;
	dpsi -= r;
	dpsi*=psi;
	dpsi *= 4.0;
	psi *= psi;
	dpsi *=psi;
	psi *= psi;
	psi*=r;
	return (psi,dpsi)
end


# function radius(x::Union{Array{Float64,2},Array{Float32,2}},eps::Float64 = 1e-3)
# t = vec(sum(x.^2,2));
# return (sqrt.(t + eps),t);
# end

# function radius(x::Union{Vector{Float64},Vector{Float32}},eps::Float64 = 1e-3)
# t = dot(x,x);
# return sqrt(t + eps);
# end

function radiust(t::Float64,eps::Float64 = 1e-3)
return sqrt(t + eps);
end

export splitTheta,wrapTheta
function splitTheta(theta::Array)
	n = div(length(theta),5);
	if 5*n != length(theta)
		error("Inappropriate length of theta");
	end
	alpha = theta[1:5:end];
	beta = theta[2:5:end];
	Xs = [theta[3:5:end] theta[4:5:end] theta[5:5:end]];
	return (alpha,beta,Xs)
end

function wrapTheta( alpha:: Array{Float64},	beta:: Array{Float64}, Xs:: Array{Float64})
	theta = zeros(5*length(alpha));
	theta[1:5:end] = alpha;
	theta[2:5:end] = beta;
	theta[3:5:end] = Xs[:,1];
	theta[4:5:end] = Xs[:,2];
	theta[5:5:end] = Xs[:,3];
	return theta
end



