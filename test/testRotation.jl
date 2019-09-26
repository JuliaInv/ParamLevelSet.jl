using ParamLevelSet
using LinearAlgebra
using Test
using jInv.Mesh

plotting = false;
if plotting
	using jInvVisPyPlot
	using PyPlot
	close("all");
end

#export getDiamondModel,getBallModel ;

function getDiamondModel(n::Array{Int64})
n_tup = tuple(n...);
n = collect(n_tup);
u = zeros(Float32,n_tup);
X,Y,Z = meshgrid(-div(n[1],2):div(n[1],2),-div(n[2],2):div(n[2],2),-div(n[3],2):div(n[3],2))
I = (abs.(X)+abs.(Y)+abs.(Z)) .<= div(n[1],4);
u[I] .= 1.0;
return u;
end

function getBallModel(n::Array{Int64})
n_tup = tuple(n...);
n = collect(n_tup);
u = zeros(Float32,n_tup);
X,Y,Z = meshgrid(-div(n[1],2):div(n[1],2),-div(n[2],2):div(n[2],2),-div(n[3],2):div(n[3],2))
I = (sqrt(X.^2+Y.^2 + Z.^2)) .<= div(n[1],4);
u[I] .= 1.0;
return u;
end




println("Rotation operator transpose test");
n = [33,33,33];

u = getDiamondModel(n);
u = convert(Array{Float64}, u)

if plotting
	figure()
	plotModel(u)
end

theta_phi = deg2rad.([37.0;24.0]);
display(theta_phi)
b = [1.0;2.0;3.0];



# RT = generateSamplingMatrix(u,theta_phi,1,1)
# v = RT'*u[:]
# println("The norm is:")
# println(norm(RT*v - u[:]))

v, = rotateAndMove3D(u,theta_phi,b,false);

if plotting
	figure()
	plotModel(v)
end
#uu = rotateAndMove3DTranspose(v,theta_phi,b);
uu, = rotateAndMove3D(v,theta_phi,b,true);
if plotting
	figure()
	plotModel(uu)
end

@test abs(dot(v,v) - dot(uu,u)) < 1e-5
@test norm(uu[:]-u[:])/norm(u[:]) < 1.0

