
export heavySide
function heavySide(x,c = 0.15,delta = 0.1,epsilon = 0.1*delta)
a = c - delta;
b = c + delta;

y = zeros(eltype(x),size(x));
dy = zeros(eltype(x),size(x));
I = (x .>= (a-epsilon)) .& (x .<= (a+epsilon));
xI = x[I];
height = 1.0/(b-a);
t = (xI .- a .+ epsilon).*((0.25*height)/epsilon)

dy[I] = 2.0*t;
y[I] = (xI .- a .+ epsilon).*t;

I = (x .> a + epsilon) .& (x .<= b - epsilon);
y[I] .= height*(x[I].-a);
dy[I] .= height;
I = (x .> b+epsilon);
y[I] .= 1.0;
dy[I] .= 0.0;
I = (x .> b-epsilon) .& (x .<= b+epsilon);
xI = x[I];
t = (xI.-(b+epsilon)).*(-(0.25*height)/epsilon);
y[I] = (xI.-(b+epsilon)).*t .+ 1.0;
dy[I] = 2.0.*t;
return y,Diagonal(dy[:]);
end

export heavySide!
function heavySide!(x::Union{Array{Float64},Array{Float32}},y::Union{Array{Float64},Array{Float32}},c::Float64 = 0.15,delta::Float64 = 0.1,epsilon::Float64 = 0.1*delta)
a = c - delta;
b = c + delta;
height = 1.0/(b-a);
temp = ((0.25*height)/epsilon);
tmp2 = 0.0;
# y = zeros(size(x));
y[:] .= 0.0;
t1 = a-epsilon;
t2 = a+epsilon;
t3 = b-epsilon;
t4 = b+epsilon;
for k=1:length(x)
	xk = x[k];
	if xk <= t1
		x[k] = 0.0;
	elseif xk >= t4
		x[k] = 1.0;
	elseif xk >= t2 && xk <=t3
		x[k] = height*(xk-a);
		y[k] = height;
	elseif xk >= t1 && xk <=t2
		tmp2 = (xk - t1)*temp;
		x[k] = (xk - t1);
		x[k] *= tmp2;
		y[k] = 2.0;
		y[k] *= tmp2;
	elseif xk >= t3 && xk <=t4
		tmp2 = (t4 - xk)*temp;
		x[k] = (xk-t4).*tmp2 + 1.0;
		y[k] = 2.0*tmp2;
	end
end
return y;
end











