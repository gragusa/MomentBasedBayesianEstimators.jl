module MomentBasedBayesianEstimators
using Reexport
@reexport using MomentBasedEstimators
@reexport using Divergences
import MomentBasedEstimators: MinimumDivergenceProblem

immutable BayesianMomentModel{F, R <: MinimumDivergenceProblem}
    f!::Function
    prior!::Function
    data::F
    m::R
    p::Array{Float64, 1}
end


flatprior!(a, theta) = return a

function BayesianMomentModel(f!::Function, prior!::Function, m::MinimumDivergenceProblem, data)
    p = zeros(1)
    BayesianMomentModel(f!, prior!, data, m, p)
end

function posterior(b::BayesianMomentModel, theta::Vector{Float64})
    ## f! is a mutating function
    b.f!(b.m.e.mm.S, theta, b.data)
    solve!(b.m)
    sum(log(b.m.m.inner.x::Array{Float64, 1})) + b.prior!(b.p, theta)
end

export BayesianMomentModel, flatprior!, posterior

# package code goes here

end # module
