using CUDA

struct Graph
    operations::Vector{Any}
end

struct Tensor{T,N}
    value::Array{T,N}
    grad::Array{T,N}
    graph::Graph
end

struct MulOp{T,N}
    input_a::Tensor{T,N}
    input_b::Tensor{T,N}
    output::Tensor{T,N}
end

struct CuMulOp{T,N}
    input_a::CuArray{T,N}
    input_b::CuArray{T,N}
    output::CuArray{T,N}
end

if CUDA.functional()
    function mul(input_a::CuArray{T,N}, input_b::CuArray{T,N}) where {T,N}
        @assert size(input_a) == size(input_b)
        output = CUDA.zeros(T, size(input_a))
        push!(graph.operations, CuMulOp(input_a, input_b, output))
        return output
    end

    function forward!(op::CuMulOp)
        @. op.output = op.input_a * op.input_b
        return nothing
    end

    function backward!(op::CuMulOp)
        input_a = op.input_a
        ∂input_a = CUDA.zeros(eltype(input_a), size(input_a))

        input_b = op.input_b
        ∂input_b = CUDA.zeros(eltype(input_b), size(input_b))

        ∂output  = CUDA.zeros(eltype(op.output), size(op.output))

        @. ∂input_a += ∂output * input_b
        @. ∂input_b += ∂output * input_a

        return nothing
    end
else
    function mul(input_a::Tensor{T,N}, input_b::Tensor{T,N}) where {T,N}
        @assert size(input_a.value) == size(input_b.value)
        @assert input_a.graph == input_b.graph
        graph = input_a.graph

        output = Tensor(zeros(T, size(input_a.value)), zeros(T, size(input_a.value)), graph)

        push!(graph.operations, MulOp(input_a, input_b, output))
        return output
    end

    function forward!(op::MulOp)
        @. op.output.value = op.input_a.value * op.input_b.value
        return nothing
    end

    function backward!(op::MulOp)
        input_a = op.input_a.value
        ∂input_a = op.input_a.grad

        input_b = op.input_b.value
        ∂input_b = op.input_b.grad

        ∂output  = op.output.grad

        @. ∂input_a += ∂output * input_b
        @. ∂input_b += ∂output * input_a

        return nothing
    end
end
