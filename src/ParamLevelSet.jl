module ParamLevelSet
using LinearAlgebra
using SparseArrays
using jInv.Mesh
using jInv.Utils

export centerHeaviside,deltaHeaviside,getDefaultHeaviside
centerHeavySide = 0.3;
deltaHeavySide = 0.05;

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
include("RBFs_extended.jl");
include("heaviside.jl");
include("RBFRotation3D.jl");
include("SPD_regularization.jl");
include("rotation3D.jl")




end

