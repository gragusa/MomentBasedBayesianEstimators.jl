using MomentBasedBayesianEstimators
using ModelsGenerators
using ProgressMeter
using Mamba
using Base.Test
srand(1113)
#=------------------------------------------------------------------------------
IV model with a first stage F of CP/m = 10 using ∑z'(y-xβ) = 0 as condition
------------------------------------------------------------------------------=#
srand(7875)
y, x, z = ModelsGenerators.sim_iv_d01(CP = 25);
y = vec(y)

data = (y, x, z);

X = Array(Float64, size(z)...)
f!(X, theta, data) = broadcast!(*, X, data[3], data[1]-data[2]*theta)
m = MinimumDivergenceProblem(X, zeros(5), div = ReverseKullbackLeibler());

bmm = BayesianMomentModel(f!, flatprior!, m, data)
#=
MCMC simulations
=#
n = 6000
burnin = 2000
sim = Chains(n + burnin, 1, names = ["Θ"]);
theta = RWMVariate([0.0]);

postf(theta) = posterior(bmm, theta)[1];

p = Progress(n, 1)
for i in 1:n + burnin
  rwm!(theta, [.5], postf, proposal = Normal)
  sim[i, :, 1] = [theta;]
  next!(p)
end
sim = sim[burnin+1:end,]
describe(sim)
p = plot(sim);
draw(p, filename="elpost_ivmom.svg")

#=------------------------------------------------------------------------------

------------------------------------------------------------------------------=#
function f!(X, θ, data)
    delta = θ[1:5]
    theta = θ[6]
    X[:] = [data[3].*(data[1]-data[3]*delta.*theta) data[3].*(data[2]-data[3]*delta)]
end

_logpostb(beta) = posterior(bmm, [delta; beta])[1]
_logpostd(delta) = posterior(bmm, [delta; beta])[1]

G = Array(Float64, 100, 10)
m = MinimumDivergenceProblem(G, zeros(10), div = ReverseKullbackLeibler())
bmm = BayesianMomentModel(f!, flatprior!, m, data)

n = 6000
burnin = 2000
sim_lik = Chains(n + burnin, 6, names = ["γ₁", "γ₂", "γ₃", "γ₄", "γ₅", "Θ"])
beta  = AMWGVariate([0.01])
delta = AMWGVariate(ones(5)*0.3)
p = Progress(n, 1)
for i in 1:n + burnin
    amwg!(beta, [.1], _logpostb, batchsize = 50, adapt = (i <= burnin))
    amwg!(delta, .1*ones(5), _logpostb, batchsize = 50, adapt = (i <= burnin))
    sim_lik[i, :, 1] = [delta; beta]
    next!(p)
end
describe(sim)

sim_lik2 = Chains(6000, 2, names = ["||δ||", "Θ"]);
sim_lik2[:,1,1] = mapslices(u -> norm(u), sim_lik.value[2001:8000,1:5,1], 2)
sim_lik2[:,2,1] = sim_lik.value[2001:8000,6,1]

p = plot(sim_lik2)
draw(p, filename="el_likmom.svg")

p = contourplot(sim_lik2, bins = 40);
draw(p, filename="el_likmom_contour.svg")

#=------------------------------------------------------------------------------

------------------------------------------------------------------------------=#

function jeffreyprior!(a, theta)
    a[1] = norm(theta[1:5])*(1+theta[6])^.5
    a
end

bmm = BayesianMomentModel(f!, jeffreyprior!, m, data)
_logpostb(beta) = posterior(bmm, [delta; beta])[1]
_logpostd(delta) = posterior(bmm, [delta; beta])[1]

n = 6000
burnin = 2000
sim_likj = Chains(n + burnin, 6, names = ["γ₁", "γ₂", "γ₃", "γ₄", "γ₅", "Θ"])
beta  = AMWGVariate([0.01])
delta = AMWGVariate(ones(5)*0.3)
p = Progress(n, 1)
for i in 1:n + burnin
    amwg!(beta, [.1], _logpostb, batchsize = 50, adapt = (i <= burnin))
    amwg!(delta, .1*ones(5), _logpostb, batchsize = 50, adapt = (i <= burnin))
    sim_likj[i, :, 1] = [delta; beta]
    next!(p)
end
describe(sim)

sim_likj2 = Chains(6000, 2, names = ["||δ||", "Θ"]);
sim_likj2[:,1,1] = mapslices(u -> norm(u), sim_likj.value[2001:8000,1:5,1], 2)
sim_likj2[:,2,1] = sim_likj.value[2001:8000,6,1]

p = plot(sim_likj2)
draw(p, filename="el_likmom_jeffrey.svg")

p = contourplot(sim_likj2, bins = 40);
draw(p, filename="el_likmom_jeffrey_contour.svg")
