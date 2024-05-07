module ParamLevelSet
using LinearAlgebra
using SparseArrays
using jInv.Mesh
using jInv.Utils
using NearestNeighbors

export centerHeaviside,deltaHeaviside,getDefaultHeaviside,getDefaultHeavyside
centerHeavySide = 0.3;
deltaHeavySide = 0.05;

function getDefaultHeavyside()
	return (u,dsu)->heaviside!(u,dsu,ParamLevelSet.centerHeavySide,ParamLevelSet.deltaHeavySide);
end

function getDefaultHeaviside()
	return (u,dsu)->heaviside!(u,dsu,ParamLevelSet.centerHeavySide,ParamLevelSet.deltaHeavySide);
end

## Other things to include: bf type in psi.
## Thresholding parameters for the Jacobians.

function getIdentityMat(n::Int64)
return Matrix(1.0I,n,n);

end
include("SpMatBuilder.jl");
include("RBFs.jl");
include("MeshFreeRBFs.jl");
include("RBFs_extended.jl");
include("heaviside.jl");
include("RBFRotation3D.jl");
include("SPD_regularization.jl");
include("rotation3D.jl")




end

