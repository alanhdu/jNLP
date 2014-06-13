type Counter <: Associative
    counts::Dict{Any, Int}
    tokens::Int
    types::Int
end

Counter() = Counter(Dict{Any, Int}(), 0, 0)

# is there some kind of inheritance (so Counter can inherit from Dict)?
function Counter(seq)
    d = (Any=>Int)[c=>count(x -> isequal(x,c), seq) for c in unique(seq)]
    n = length(seq)
    b = length(unique(seq))
    return Counter(d, n, b)
end
keys(xs::Counter) = Base.keys(xs.counts)
haskey(xs::Counter, key) = Base.haskey(xs.counts, key)
values(xs::Counter) = Base.values(xs.counts)
get(xs::Counter, key, default) = Base.get(xs.counts, key, default)
getindex(xs::Counter,key) = Base.get(xs.counts, key, 0)
setindex!(xs::Counter,key, val) = setindex!(xs.counts, key, val)

hapax(xs::Counter) = filter(k->xs[k]==1,keys(xs))
function update!(xs::Counter, seq)
    xs.tokens += length(seq)
    for c in unique(seq)
        if haskey(xs, c)
            xs[c] += count(x->isequal(x,c), seq)
        else
            xs[c] = count(x->isequal(x,c), seq)
            xs.types += 1
        end
    end
end

abstract Smoother
type Unseen
end
logprob(x::Smoother, obs) = log(prob(x, obs))
mass(x::Smoother) = sum([prob(x, s) for s in keys(x.freq)])
keys(x::Smoother) = keys(x.freq)

function generate(x::Smoother) 
    r = rand() * mass(x)
    for s in keys(x)
        r -= prob(x, s)
        if r <= 0.001    # roundoff error
            return s
        end
    end
end

type MLESmooth <: Smoother
    freq::Counter
end
prob(x::MLESmooth, obs) = x.freq[obs] / x.freq.tokens

type LidstoneSmooth <: Smoother
    freq::Counter
    gamma::Real
end
LidstoneSmooth(xs::Counter) = LidstoneSmooth(xs, 1) # default to Laplace
prob(x::LidstoneSmooth, obs) = (x.freq[obs] + x.gamma) / (x.freq.tokens + x.freq.types * x.gamma)

type HeldoutSmooth <: Smoother
    freq::Counter
    held::Counter
end
# TODO edit julia so that collect is unnecessary
_Tr(x::HeldoutSmooth, r::Int)=sum([x.held[c] for c in collect(filter(y->x.freq[y]==r, keys(x.freq)))])
_Nr(x::HeldoutSmooth, r::Int)=count(y->x.freq[y]==r, keys(x.freq))

function prob(x::HeldoutSmooth, obs)
    if haskey(x.freq, obs)
        return _Tr(x, x.freq[obs]) / (_Nr(x, x.freq[obs]) * x.held.tokens)
    else
        return 0
    end
end

type CrossValidateSmooth <: Smoother
    freqs::Array{Counter, 1}
    probs::Array{HeldoutSmooth, 1}
    function CrossValidateSmooth(xs...)
        probs = Array(HeldoutSmooth, 0)
        for x in xs
            for y in xs
                if !is(x,y)
                    push!(probs, HeldoutSmooth(x, y))
                end
            end
        end
        return new(collect(xs), probs)
    end
end
keys(x::CrossValidateSmooth) = mapreduce(y->Set(keys(y)...), union, x.freqs)
prob(x::CrossValidateSmooth, obs) = sum([prob(y, obs) for y in x.probs]) / length(x.probs)

type GoodTuringSmooth <: Smoother
    freq::Counter
    interpolater::Function
end
GoodTuringSmooth(x::Counter) = GoodTuringSmooth(x, identity)
rStar(x::GoodTuringSmooth, r) = (r+1) * interpolater(r+1) / interpolater(r)
prob(x::GoodTuringSmooth, obs) = (haskey(x.freq, obs) ? rStar : hapax(x.freq)) / x.freq.tokens

type SimpleGoodTuringSmooth <: Smoother
    freq::Counter
    a::Real
    b::Real
end
function SimpleGoodTuringSmooth(xs::Counter)
    rank, Ns = zip(enumerate(reverse(sort(collect(values(xs)))))...)
end
