export getRotate3D,rotateAndMove3D,rotateAndMove3DTranspose,smooth3D,rotateSensitivity3D

function loc2cs3D(loc1::Union{Int64,Array{Int64}},loc2::Union{Int64,Array{Int64}},loc3::Union{Int64,Array{Int64}},n::Array{Int64,1})
@inbounds cs = loc1 .+ (loc2.-1)*n[1] .+ (loc3.-1)*n[1]*n[2];
return cs;
end


function getRotate3D(theta_rad::Float64,phi_rad::Float64)
## theta rotates on the x-y plain (or x1-x2)
## phi rotates on the x-z plain (or x1-x3).
ct = cos(theta_rad); st = sin(theta_rad);
cp = cos(phi_rad); sp = sin(phi_rad);
Rxy = [ct -st 0.0; st ct 0.0; 0.0 0.0 1.0];
Rxz = [cp 0.0 sp; 0.0 1.0 0.0; -sp 0.0 cp];
R   = Rxz*Rxy;
return R;
end

function rotateAndMove3D(u::Array{Float64,3},theta_phi_rad::Vector{Float64},b::Vector{Float64},doTranspose=false,v::Array = copy(u),XT::Array = zeros(0),XTT = zeros(0))
###
# b here is in pixels - not in Mesh units (no mesh here...)
###
n = collect(size(u));
v[:] .= 0.0;
R = getRotate3D(theta_phi_rad[1],theta_phi_rad[2]);
invR = inv(R);
midphalf = n./2.0 .+ 0.5;

midphalf_plus_b = midphalf .+ b;

maxU = maximum(u);
thresh = 1e-4*maxU;

xt = zeros(3);
x = zeros(3);


if length(XT)==0
	# println("Building XT");
	XT = zeros(3,prod(n));
	for i3 = 1:size(u,3)
		for i2 = 1:size(u,2)
			for i1 = 1:size(u,1)
				@inbounds xt[1] = i1; xt[2] = i2; xt[3] = i3;
				# x = xt - midphalf_plus_b;
				BLAS.axpy!(-1.0,midphalf_plus_b,xt); # xt = xt - midphalf_plus_b;
				ii = loc2cs3D(i1,i2,i3,n);
				@inbounds XT[:,ii] = xt;
			end
		end
	end
	XTT = zeros(3,prod(n));
end

iii=1;
# gemm!(tA, tB, alpha, A, B, beta, C)
XTT[1,:] .= midphalf[1];
XTT[2,:] .= midphalf[2];
XTT[3,:] .= midphalf[3];
BLAS.gemm!('N','N',1.0,invR,XT,1.0,XTT);
# I think that here we assume that the domain is square to simplify.
II = findall(vec(sum((XTT .> 0.5) .& (XTT .< n[1]),dims=1).>=3));

LOC = round.(Int64,XTT[:,II]);
LOC = loc2cs3D(LOC[1,:],LOC[2,:],LOC[3,:],n);

if !doTranspose
	v[II] = u[LOC];
else
	# The following two options are actually different in terms of repeatitions:
	# v[LOC] .+= u[II];
	for ii=1:length(LOC)
		v[LOC[ii]] += u[II[ii]];
	end
end

# the code below is an altenative to the code above.
# for i3 = 1:size(u,3)
	# for i2 = 1:size(u,2)
		# for i1 = 1:size(u,1)
			# @inbounds xt[1] = i1; xt[2] = i2; xt[3] = i3;
			# x = xt - midphalf_plus_b;
			# BLAS.axpy!(-1.0,midphalf_plus_b,xt); # xt = xt - midphalf_plus_b;
			# BLAS.gemv!('N',1.0,invR,xt,0.0,x); # x = 0.0*x + invR*xt;
			# BLAS.axpy!(1.0,midphalf,x); # x = xt + midphalf;
			# @inbounds x[:] = XTT[:,iii];
			# iii+=1;


			# @inbounds if x[1] < 0.5 || (x[1] > n[1]) || (x[2] < 0.5) || (x[2] > n[2])|| x[3] < 1 || x[3] > n[3]
				# continue;
			# end
			# ii = round.(Int64,x);
			# @inbounds v[i1,i2,i3] = u[ii[1],ii[2],ii[3]];
		# end
	# end
# end
return v,XT,XTT;
end

function rotateAndMove3DTranspose(v::Array,theta_phi_rad::Array{Float64},b::Array{Float64})
###
# b here is in pixels - not in Mesh units (no mesh here...)
###
n = collect(size(v));
u = zeros(eltype(v),size(v));
R = getRotate3D(theta_phi_rad[1],theta_phi_rad[2]);
invR = inv(R);
mid = n/2.0;
maxU = maximum(v);
thresh = 1e-4*maxU;
for i3 = 1:size(v,3)
	for i2 = 1:size(v,2)
		for i1 = 1:size(v,1)
			@inbounds vi = v[i1,i2,i3];
			if abs(vi) > thresh
				x = [i1;i2;i3] .- (mid .+ 0.5 ) .- b; ### here we actually do the inverse operation in a forward way
				x = invR*x;
				x .+= mid;
				@inbounds if x[1] <=  0.5 || (x[1] >= n[1]-0.5) || (x[2] <= 0.5) || (x[2] >= n[2]-0.5)|| x[3] <= 0.5 || x[3] >= n[3]-0.5
					warn("Rotation3D: rotation went out of bounds. Please increase zero padding.");
					continue;
				end
				x .+= 0.5;
				xloc = convert(Array{Int64},round.(x));
				@inbounds u[xloc[1],xloc[2],xloc[3]] += vi;
			end
		end
	end
end
return u;
end


function smooth3D(v::Array,numFilters = 1)
N = collect(-1:1);
E3 = ones(3,3,3);
E2 = [1.0 ; 2.0 ; 1.0]*[1.0 ; 2.0 ; 1.0]';
E3[:,:,1] = E2;
E3[:,:,2] = 2*E2;
E3[:,:,3] = E2;
E = E ./ sum(E[:]);

for k = 1:2
	for i3 = 2:size(u,3)-1
		for i2 = 2:size(u,2)-1
			for i1 = 2:size(u,1)-1
				v[i1, i2, i3] = sum(E.*v[i1 + N, i2 + N, i3 + N]);
			end
		end
	end
end
return v;
end


# function rotateSensitivity3D(u::Array,dtheta::Array{Float64},db::Array{Float64},b::Array{Float64})
# n = collect(size(u));
# h = ones(size(n));
# v = zeros(eltype(u),size(u));
# mid = n/2.0 + b;
# dtheta = deg2rad(dtheta);
# for i3 = 2:size(u,3)-1
	# for i2 = 2:size(u,2)-1
		# for i1 = 2:size(u,1)-1
			# du1 = (u[i1+1,i2,i3] - u[i1-1,i2,i3])/2;
			# du2 = (u[i1,i2+1,i3] - u[i1,i2-1,i3])/2;
			# du3 = (u[i1,i2,i3+1] - u[i1,i2,i3-1])/2;
			# x = [i1,i2,i3] - (mid + 0.5);
			# v[i1,i2,i3] = db[1]*du1 + db[2]*du2 + db[3]*du3 + (dtheta[1]*x[2] + dtheta[2]*x[3])*du1 - dtheta[1]*x[1]*du2 - dtheta[2]*x[1]*du3;
		# end
	# end
# end
# return v;
# end
