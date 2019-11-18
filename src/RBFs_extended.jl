export getA
function computeFunc_ellipsoids!(m::Vector{Float64},k::Int64,Xc,h::Vector{Float64},n::Vector{Int64},
		u::Vector{Float64},xmin::Vector{Float64},beta::Float64)
numParamOfRBF = 10;
(x1,x2,x3) = getXt(k,m,numParamOfRBF);
i = round.(Int64,([x1;x2;x3] - xmin)./h .+ 0.5);
# ## here we take a reasonable box around the bf. 
# ## The constant depends on the type of RBF that we take.
(a1,a2,a3,a4,a5,a6) = getA(k,m);
boxL = ceil.(Int64,0.9./(h*beta));
imax = min.(i + boxL,n);
imin = max.(i - boxL,[1;1;1]);
alpha = getAlpha(k,m,numParamOfRBF);


iishift2 = 0;
for l = imin[3]:imax[3]
	@inbounds iishift3 = (l-1)*n[1]*n[2];
	for j = imin[2]:imax[2]
		@inbounds iishift2 = iishift3 + (j-1)*n[1];
		for q = imin[1]:imax[1]
			ii = iishift2 + q;	
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
end
end


function updateJacobian_ellipsoids!(k::Int64,m::Vector{Float64},dsu::Vector{Float64},h::Vector{Float64},n::Vector{Int64},Jbuilder::SpMatBuilder,u::Vector{Float64},iifunc::Function,Xc,xmin::Vector{Float64},beta::Float64)
numParamOfRBF = 10;
(x1,x2,x3) = getXt(k,m,numParamOfRBF);
alpha = getAlpha(k,m,numParamOfRBF);
(a1,a2,a3,a4,a5,a6) = getA(k,m);

i = round.(Int64,([x1;x2;x3] - xmin)./h .+ 0.5);
# ## here we take a reasonable box around the bf. 
# ## The constant depends on the type of RBF that we take.
boxL = ceil.(Int64,0.9./(h*beta));
imax = min.(i + boxL,n);
imin = max.(i - boxL,[1;1;1]);


offset = convert(Int64,(k-1)*numParamOfRBF + 1);
md = 1e-3*maximum(abs.(dsu));
temp = 0.0;

iishift2 = 0;
for l = imin[3]:imax[3]
	@inbounds iishift3 = (l-1)*n[1]*n[2];
	for j = imin[2]:imax[2]
		@inbounds iishift2 = iishift3 + (j-1)*n[1];
		for q = imin[1]:imax[1]
			ii = iishift2 + q;
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
					ii = iifunc(ii);
					setNext!(Jbuilder,ii,offset,psi);
					temp *= alpha;
					temp /= radii;
					temp *= dpsi;
					psi = z1; psi*=temp;
					setNext!(Jbuilder,ii,offset+1,0.5*z1*psi);
					setNext!(Jbuilder,ii,offset+2,z2*psi);
					setNext!(Jbuilder,ii,offset+3,z3*psi);
					psi = z2; psi*=temp;z2*=0.5;z2*=psi;
					setNext!(Jbuilder,ii,offset+4,z2);
					setNext!(Jbuilder,ii,offset+5,z3*psi);
					setNext!(Jbuilder,ii,offset+6,0.5*z3*z3*temp);
					Az1*=temp;Az2 *= temp; Az3 *= temp; 
					setNext!(Jbuilder,ii,offset+7,Az1);
					setNext!(Jbuilder,ii,offset+8,Az2);
					setNext!(Jbuilder,ii,offset+9,Az3);			
				end
			end
		end
	end
end 
end

function getA(k::Int64,m::Vector{Float64})
	return (m[10*(k-1) + 2],m[10*(k-1) + 3],m[10*(k-1) + 4],m[10*(k-1) + 5],m[10*(k-1) + 6],m[10*(k-1) + 7]);
end
export idx_trilA
function idx_trilA()
	return [1;2;3;5;6;9];
end


## A:
# a1 a2 a3
# a2 a4 a5
# a3 a5 a6



function mulA(a1::Float64,a2::Float64,a3::Float64,a4::Float64,a5::Float64,a6::Float64,x1::Float64,x2::Float64,x3::Float64)
t = 0.0;
a1 *= x1; t = a2; t*=x2; a1+=t; t=a3; t*=x3; a1+=t;
a4 *= x2; a2*=x1; a4+=a2;  t = a5; t*=x3; a4+=t; ## a2 is now run over.
a6 *= x3; a3*=x1; a6 += a3; a5*=x2; a6 += a5;
return (a1,a4,a6);
end


# A = randn(3,3); A = A'*A + eye(3); #L = 0.5*(L+L');
# x = randn(3);
# dA = 0.005*randn(3,3); dA = 0.5*(dA+dA');

# function forward(x,A)
# return dot(x,A*x);
# end

# function der(x,A)
# C = (x*x');
# return C;
# end


# u0 = forward(x,A);
# for k=1:5
	# dA.*=0.5;
	# u = forward(x,A+dA);
	# ut = u0 + vec(der(x,A))'*vec(dA);
	# println("norm(ut-u0): ",abs(u-u0),", norm(ut - u0 - J0*dm): ",norm(u - ut));
# end



#########################################################################################################
#########################################################################################################
#########################################################################################################
# In the old code below we use a representation using a Cholesky factor.
# This code is still with Ihuge 
# If no rotations are applied, using L frees us from the SPD regularization.

# function computeFuncAndUpdateIhuge_ellipsoids!(m::Vector{Float64},k::Int64,Xc::Array{Float64},h::Vector{Float64},n::Vector{Int64},
		# Ihuge::Vector{Int32},Istarts::Vector{Int64},u::Vector{Float64},xmin::Vector{Float64},ii_count_huge::Int64)
# numParamOfRBF = 10;
# (x1,x2,x3) = getXt(k,m,numParamOfRBF);
# i = round.(Int64,(getX(k,m,numParamOfRBF) - xmin)./h + 0.5);
# # ## here we take a reasonable box around the bf. 
# # ## The constant depends on the type of RBF that we take.
# (l1,l2,l3,l4,l5,l6) = getL(k,m);
# L = [l1 0.0 0.0 ; l2 l4 0.0 ; l3 l5 l6];
# beta = 0.5*sqrt(maximum(eig(L*L')[1]));
# boxL = ceil.(Int32,0.85./(h*beta));
# imax = min.(i + boxL,n);
# imin = max.(i - boxL,[1;1;1]);
# alpha = getAlpha(k,m,numParamOfRBF);

# for l = imin[3]:imax[3]
	# iishift3 = (l-1)*n[1]*n[2];
	# for j = imin[2]:imax[2]
		# iishift2 = iishift3 + (j-1)*n[1];
		# for q = imin[1]:imax[1]
			# ii = iishift2 + q;
			# nX = 0.0;
			# @inbounds y1 = Xc[ii,1]-x1;
			# @inbounds y2 = Xc[ii,2]-x2;
			# @inbounds y3 = Xc[ii,3]-x3;
			# y1,y2,y3 = mulLT(l1,l2,l3,l4,l5,l6,y1,y2,y3);
			# y1*=y1;y2*=y2;y3*=y3; 
			# nX+=y1;nX+=y2;nX+=y3;

			# if (nX <= 0.77) # 0.77~0.875^2
				# argii = radiust(nX);
				# argii = psi1(argii);
				# argii *= alpha;
				# @inbounds u[ii] += argii;
				# Ihuge[ii_count_huge] = ii;
				# ii_count_huge += 1;
			# end
		# end
	# end
# end
# Istarts[k+1] = ii_count_huge;
# return ii_count_huge;
# end



# function updateJacobianArrays_ellipsoids!(k::Int64,m::Vector{Float64},curr::Int64,dsu::Vector{Float64},Is::Vector{Int32},Js::Vector{Int32},Vs::Vector{Float64},u::Vector{Float64},Ihuge::Vector{Int32},Istarts::Vector{Int64},iifunc::Function,Xc::Array{Float64})
# numParamOfRBF = 10;
# (x1,x2,x3) = getXt(k,m,numParamOfRBF);
# alpha = getAlpha(k,m,numParamOfRBF);
# (l1,l2,l3,l4,l5,l6) = getL(k,m);

# offset = convert(Int32,(k-1)*numParamOfRBF + 1);
# temp = 0.0;
# for iih = (Istarts[k]):(Istarts[k+1]-1)
	# @inbounds ii = Ihuge[iih];
	# @inbounds temp = dsu[ii];
	# @inbounds z1 = x1 - Xc[ii,1]; 
	# @inbounds z2 = x2 - Xc[ii,2]; 
	# @inbounds z3 = x3 - Xc[ii,3];
	# (Ltz1,Ltz2,Ltz3) = mulLT(l1,l2,l3,l4,l5,l6,z1,z2,z3);
	# radii = Ltz1*Ltz1 + Ltz2*Ltz2 + Ltz3*Ltz3;
	# # radii = z1*z1 + z2*z2 + z3*z3;
	# radii = radiust(radii);
	# psi,dpsi = dpsi1_t(radii);
	# psi*=temp;
	# @inbounds Vs[curr] = psi;
	# temp *= alpha;
	# temp /= radii;
	# temp *= dpsi;
	# Ltz1*=temp; Ltz2*=temp; Ltz3*=temp;
	# @inbounds Vs[curr+1] = Ltz1*z1; 
	# @inbounds Vs[curr+2] = Ltz1*z2;
	# @inbounds Vs[curr+3] = Ltz1*z3;
	# @inbounds Vs[curr+4] = Ltz2*z2;
	# @inbounds Vs[curr+5] = Ltz2*z3;
	# @inbounds Vs[curr+6] = Ltz3*z3;
	# (Az1, Az2, Az3) = mulL(l1,l2,l3,l4,l5,l6,Ltz1,Ltz2,Ltz3);
	# # Az1 = z1*temp;Az2 = z2*temp;Az3 = z3*temp; 
	# @inbounds Vs[curr+7] = Az1;
	# @inbounds Vs[curr+8] = Az2;
	# @inbounds Vs[curr+9] = Az3;
	# ii = iifunc(ii);
	# for p = 0:(numParamOfRBF-1)
		# @inbounds Is[curr+p] = ii;
		# @inbounds Js[curr+p] = offset + Int32(p);
	# end				
	# curr += numParamOfRBF;				
# end
# return curr;
# end 

# m is ordered as alpha, l1, l2, l3, l4, l5, l6, x1,x2,x3

## L:
# L1 0 0
# L2 L4 0
# L3 L5 L6


# function getL(k::Int64,m::Vector{Float64})
	# return (m[10*(k-1) + 2],m[10*(k-1) + 3],m[10*(k-1) + 4],m[10*(k-1) + 5],m[10*(k-1) + 6],m[10*(k-1) + 7]);
# end

# function mulL(L1::Float64,L2::Float64,L3::Float64,L4::Float64,L5::Float64,L6::Float64,x1::Float64,x2::Float64,x3::Float64)
# L1 *= x1;
# L4 *= x2; L2*=x1; L4+=L2;
# L3 *= x1; L5*=x2; L6*=x3; L6+=L5; L6+=L3;
# return (L1,L4,L6);
# end

# function mulLT(L1::Float64,L2::Float64,L3::Float64,L4::Float64,L5::Float64,L6::Float64,x1::Float64,x2::Float64,x3::Float64)
# L1 *= x1; L2*=x2; L3*=x3; L1 += L2; L1+=L3;
# L4 *= x2; L5*=x3; L4+=L5;
# L6*=x3;
# return (L1,L4,L6);
# end




# L = randn(3,3); #L = 0.5*(L+L');
# x = randn(3);
# dL = 0.5*randn(3,3); #dL = 0.5*(dL+dL');

# function forward(x,L)
# return dot(L'*x,L'*x);
# end

# function der(x,L)
# C = 2.0*(L'*x)*x';
# return C';
# end


# u0 = forward(x,L);
# for k=1:5
	# dL.*=0.5;
	# u = forward(x,L+dL);
	# ut = u0 + vec(der(x,L))'*vec(dL);
	# # ut = u0 + 2.0*dot(L'*x,dL'*x);
	# # ut = u0 + 2.0*trace(dL'*x*x'*L);
	# # ut = u0 + 2.0*vecdot(dL,x*x'*L);
	# println("norm(ut-u0): ",abs(u-u0),", norm(ut - u0 - J0*dm): ",norm(u - ut));
# end





