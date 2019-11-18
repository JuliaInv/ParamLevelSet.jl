using Test;
using LinearAlgebra;
using SparseArrays;
using ParamLevelSet;
using Random;
n = 10;
B = getSpMatBuilder(Int64,Float64,n, n, 30);

for k=1:10
	setNext!(B,k,k,1.0);
end
II = getSparseMatrix(B);
@test norm(II - SparseMatrixCSC(1.0I,n,n)) < 1e-14

B.V.*=100.0;
reset!(B);


for i=1:10
	JJ = randperm(n)[1:5];
	for j = 1:5
		setNext!(B,i,JJ[j],1.0);
	end
end
@test sum(B.V[:]) == 50;
II = getSparseMatrix(B);
IIT = getSparseMatrixTransposed(B);
@test norm(II - IIT') < 1e-14

