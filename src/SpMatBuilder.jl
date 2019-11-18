
mutable struct SpMatBuilder{T,S}
    I   :: Array{T}
	J   :: Array{T}
	V   :: Array{S}
	m   :: Int64 
	n   :: Int64
	idx :: Int64
end

export getSpMatBuilder,reset!,getSparseMatrix,setNext!,getSparseMatrixTransposed;

function getSpMatBuilder(T::Type,S::Type, m::Int64, n::Int64, nnz::Int64)
	return SpMatBuilder{T,S}(ones(T,nnz),ones(T,nnz),zeros(S,nnz),m,n,1);
end

function reset!(B::SpMatBuilder, m_new::Int64 = B.m, n_new::Int64=B.n)
	B.m = m_new;
	B.n = n_new;
	B.idx = 1;
end

function getSparseMatrix(B::SpMatBuilder)
	idx = B.idx;
	B.I[idx:end] .= 1;
	B.J[idx:end] .= 1;
	B.V[idx:end] .= 0.0;
	return sparse(B.I,B.J,B.V,B.m,B.n);
end

function getSparseMatrixTransposed(B::SpMatBuilder)
	idx = B.idx;
	B.I[idx:end] .= 1;
	B.J[idx:end] .= 1;
	B.V[idx:end] .= 0.0;
	return sparse(B.J,B.I,B.V,B.m,B.n);
end



function setNext!(B::SpMatBuilder{T,S},i::T, j::T, v::S) where {T,S}
idx = B.idx;
if idx <= length(B.I)
	@inbounds B.I[idx] = i;
	@inbounds B.J[idx] = j;
	@inbounds B.V[idx] = v;
	B.idx += 1;
else
	nnz = length(B.I);
	B_new = getSpMatBuilder(T,S, B.m, B.n, 2*nnz)
	B_new.I[1:nnz] .= B.I;
	B_new.J[1:nnz] .= B.J;
	B_new.V[1:nnz] .= B.V;
	B.I = B_new.I;
	B.J = B_new.J;
	B.V = B_new.V;
	setNext!(B,i,j,v);
end
return;
end

