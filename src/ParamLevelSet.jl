module ParamLevelSet
using LinearAlgebra
using SparseArrays
using jInv.Mesh
using jInv.Utils
#using VolReconstruction.Utils

export centerHeavySide,deltaHeavySide,getDefaultHeavySide
centerHeavySide = 0.3;
deltaHeavySide = 0.05;


# function getDefaultHeavySide()
	# return (u)->heavySide(u,ParamLevelSet.centerHeavySide,ParamLevelSet.deltaHeavySide);
# end

function getDefaultHeavySide()
	return (u,dsu)->heavySide!(u,dsu,ParamLevelSet.centerHeavySide,ParamLevelSet.deltaHeavySide);
end



## Other things to include: bf type in psi.
## Thresholding parameters for the Jacobians.

function getIdentityMat(n::Int64)
return Matrix(1.0I,n,n);

end
include("RBFs.jl");
include("RBFs_extended.jl");
include("heavySide.jl");
include("RBFRotation3D.jl");
include("SPD_regularization.jl");
include("rotation3D.jl")




end

