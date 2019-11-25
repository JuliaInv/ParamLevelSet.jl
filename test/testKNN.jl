using NearestNeighbors
using Distances
using LinearAlgebra
data = rand(3, 10^4);
point = rand(3);
#println("Our point")
#println(point)
tree = BallTree(data);

t = @elapsed minimum.(data)
t = @elapsed minimum.(data)
println("Reference time: ",t)
@elapsed tree = BallTree(data);
t = @elapsed tree = BallTree(data);
println("BallTree time: ",t)

@elapsed kdtree = KDTree(data);

k = 10;
t = @elapsed idxs, dists = knn(kdtree, point, k, true);
t = @elapsed idxs, dists = knn(kdtree, point, k, true);
#println("time:",t)

# display(idxs)
# println("")
# display(dists)
# println("")

@elapsed idxs, dists = knn(tree, point, k, true)
t = @elapsed idxs, dists = knn(tree, point, k, true)
#println("time:",t)

# display(idxs)
# println("")
# display(dists)
# println("")

r = 0.02;
@elapsed idxs = inrange(tree, point, r)
t = @elapsed idxs = inrange(tree, point, r)
# println("time:",t)
# display(data[:,idxs])
#println("**************************************************************************************")
A = randn(3,3);
A = A'*A + 0.1*Matrix(1.0I,3,3);
t = @elapsed tree = BallTree(data,Mahalanobis{Float64}(A));
t = @elapsed tree = BallTree(data,Mahalanobis{Float64}(A));
println("time:",t)

t = @elapsed idxs, dists = knn(tree, point, k, true)

t = @elapsed idxs, dists = knn(tree, point, k, true)
# println("time:",t)

# display(idxs)
# println("")
# display(dists)
# println("")

r = 0.02;
@elapsed idxs = inrange(tree, point, r)
t = @elapsed idxs = inrange(tree, point, r)
# println("time:",t)
# display(data[:,idxs])