abstract type AbstractAccArray{T, N} <: DenseArray{T, N} end
# Sampler type that acts like a texture/image and allows interpolated access
abstract type AbstractSampler{T, N} <: AbstractAccArray{T, N} end

const AccVector{T} = AbstractAccArray{T, 1}
const AccMatrix{T} = AbstractAccArray{T, 2}
const AccVecOrMat{T} = Union{AbstractAccArray{T, 1}, AbstractAccArray{T, 2}}



mutable struct GPUArray{T, N, B, C} <: AbstractAccArray{T, N}
    buffer::B
    size::NTuple{N, Int}
    context::C
end


# interfaces

#=
Interface for accessing the lower level
=#

buffer(A::AbstractAccArray) = A.buffer
context(A::AbstractAccArray) = A.context
default_buffer_type(typ, context) = error("Found unsupported context: $context")

# GPU Local Memory
struct LocalMemory{T} <: AbstractAccArray{T, 1}
    size::Int
end


"""
linear index in a GPU kernel
"""
function linear_index end



#=
AbstractArray interface
=#
Base.eltype(::AbstractAccArray{T}) where {T} = T
Base.size(A::AbstractAccArray) = A.size

function Base.show(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    println(io, "GPUArray with ctx: $(context(A)): ")
    show(io, mt, Array(A))
end
function Base.showcompact(io::IO, mt::MIME"text/plain", A::AbstractAccArray)
    showcompact(io, mt, Array(A))
end

function Base.similar(x::T) where T <: AbstractAccArray
    similar(x, eltype(x), size(x))
end
function Base.similar(x::T, ::Type{ET}; kw_args...) where {T <: AbstractAccArray, ET}
    similar(x, ET, size(x); kw_args...)
end
function Base.similar(x::T, dims::NTuple{N, Int}; kw_args...) where {T <: AbstractAccArray, N}
    similar(x, eltype(x), dims; kw_args...)
end
function Base.similar(x::AbstractAccArray, ::Type{ET}, sz::NTuple{N, Int}; kw_args...) where {N, ET}
    similar(typeof(x), ET, sz, context = context(x); kw_args...)
end


using Compat.TypeUtils
function Base.similar(
        ::Type{T}, ::Type{ET}, sz::NTuple{N, Int};
        context::Context = current_context(), kw_args...
    ) where {T <: GPUArray, ET, N}
    bt = default_buffer_type(T, Tuple{ET, N}, context)
    GPUArray{ET, N, bt, typeof(context)}(sz; context = context)
end






#=
Host to Device data transfers
=#
function (::Type{A})(x::AbstractArray) where A <: AbstractAccArray
    A(collect(x))
end
function (::Type{A})(x::Array; kw_args...) where A <: AbstractAccArray
    out = similar(A, eltype(x), size(x); kw_args...)
    copy!(out, x)
    out
end
Base.convert(::Type{A}, x::AbstractArray) where {A <: AbstractAccArray} = A(x)
Base.convert(::Type{A}, x::A) where {A <: AbstractAccArray} = x

#=
Device to host data transfers
=#
function Array(device_array::AbstractAccArray{T, N}) where {T, N}
    Array{T, N}(device_array)
end
function (AT::Type{Array{T, N}})(device_array::AbstractAccArray) where {T, N}
    convert(AT, Array(device_array))
end
function (AT::Type{Array{T, N}})(device_array::AbstractAccArray{T, N}) where {T, N}
    hostarray = similar(AT, size(device_array))
    copy!(hostarray, device_array)
    hostarray
end


# Function needed to be overloaded by backends
function mapidx end
# same for mapreduce
function acc_mapreduce end


######################################
# Broadcast
include("broadcast.jl")


# we need to overload all the different broadcast functions, since x... is ambigious

# TODO check size
function Base.map!(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    broadcast!(f, A, args...)
end
function Base.map(f::Function, A::AbstractAccArray, args::AbstractAccArray...)
    broadcast(f, A, args...)
end


#############################
# reduce

# hack to get around of fetching the first element of the GPUArray
# as a startvalue, which is a bit complicated with the current reduce implementation
function startvalue(f, T)
    error("Please supply a starting value for mapreduce. E.g: mapreduce($f, $op, 1, A)")
end
startvalue(::typeof(+), T) = zero(T)
startvalue(::typeof(*), T) = one(T)
startvalue(::typeof(Base.scalarmin), T) = typemax(T)
startvalue(::typeof(Base.scalarmax), T) = typemin(T)

# TODO widen and support Int64 and use Base.r_promote_type
gpu_promote_type(op, ::Type{T}) where {T} = T
gpu_promote_type(op, ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(+), ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(*), ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(+), ::Type{T}) where {T<:Number} = typeof(zero(T)+zero(T))
gpu_promote_type(::typeof(*), ::Type{T}) where {T<:Number} = typeof(one(T)*one(T))
gpu_promote_type(::typeof(Base.scalarmax), ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(Base.scalarmin), ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(max), ::Type{T}) where {T<:Base.WidenReduceResult} = T
gpu_promote_type(::typeof(min), ::Type{T}) where {T<:Base.WidenReduceResult} = T

function Base.mapreduce(f::Function, op::Function, A::AbstractAccArray{T, N}) where {T, N}
    OT = gpu_promote_type(op, T)
    v0 = startvalue(op, OT) # TODO do this better
    mapreduce(f, op, v0, A)
end


function Base.mapreduce(f, op, v0, A::AbstractAccArray, B::AbstractAccArray, C::Number)
    acc_mapreduce(f, op, v0, A, (B, C))
end
function Base.mapreduce(f, op, v0, A::AbstractAccArray, B::AbstractAccArray)
    acc_mapreduce(f, op, v0, A, (B,))
end
function Base.mapreduce(f, op, v0, A::AbstractAccArray)
    acc_mapreduce(f, op, v0, A, ())
end

############################################
# Constructor

function Base.fill!(A::AbstractAccArray{T, N}, val) where {T, N}
    valconv = T(val)
    gpu_call(const_kernel2, A, (A, valconv, Cuint(length(A))))
    A
end
function Base.rand(::Type{T}, ::Type{ET}, size...) where {T <: AbstractAccArray, ET}
    T(rand(ET, size...))
end


############################################
# serialization

const BaseSerializer = if isdefined(Base, :AbstractSerializer)
    Base.AbstractSerializer
elseif isdefined(Base, :SerializationState)
    Base.SerializationState
else
    error("No Serialization type found. Probably unsupported Julia version")
end

function Base.serialize(s::BaseSerializer, t::T) where T <: GPUArray
    Base.serialize_type(s, T)
    serialize(s, Array(t))
end
function Base.deserialize(s::BaseSerializer, ::Type{T}) where T <: GPUArray
    A = deserialize(s)
    T(A)
end

import Base: copy!, getindex, setindex!

@inline unpack_buffer(x) = x
@inline unpack_buffer(x::AbstractAccArray) = buffer(x)
@inline unpack_buffer(x::Ref{<: AbstractAccArray}) = unpack_buffer(x[])

function to_cartesian(A, indices::Tuple)
    start = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return first(val)
        isa(val, Colon) && return 1
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    stop = CartesianIndex(ntuple(length(indices)) do i
        val = indices[i]
        isa(val, Integer) && return val
        isa(val, UnitRange) && return last(val)
        isa(val, Colon) && return size(A, i)
        error("GPU indexing only defined for integers or unit ranges. Found: $val")
    end)
    CartesianRange(start, stop)
end

crange(start, stop) = CartesianRange(CartesianIndex(start), CartesianIndex(stop))

#Hmmm... why is this not part of the Array constructors???
#TODO Figure out or issue THEM JULIA CORE PEOPLE SO HARD ... or PR? Who'd know
function array_convert(t::Type{Array{T, N}}, x::Array) where {T, N}
    convert(t, x)
end


array_convert(t::Type{Array{T, N}}, x::T) where {T, N} = [x]

function array_convert(t::Type{Array{T, N}}, x::T2) where {T, N, T2}
    arr = collect(x) # iterator
    dims = ntuple(Val{N}) do i
        ifelse(ndims(arr) >= i, size(arr, i), 1)
    end
    return reshape(map(T, arr), dims) # broadcast dims
end

for (D, S) in ((AbstractAccArray, AbstractArray), (AbstractArray, AbstractAccArray), (AbstractAccArray, AbstractAccArray))
    @eval begin
        function copy!(
                dest::$D, doffset::Integer,
                src::$S, soffset::Integer, amount::Integer
            )
            copy!(
                unpack_buffer(dest), doffset,
                unpack_buffer(src), soffset, amount
            )
        end
        function copy!(
                dest::$D{T, N}, rdest::NTuple{N, UnitRange},
                src::$S{T, N}, ssrc::NTuple{N, UnitRange},
            ) where {T, N}
            drange = crange(start.(rdest), last.(rdest))
            srange = crange(start.(ssrc), last.(ssrc))
            copy!(dest, drange, src, srange)
        end
        function copy!(
                dest::$D{T, 1}, d_range::CartesianRange{CartesianIndex{1}},
                src::$S{T, 1}, s_range::CartesianRange{CartesianIndex{1}},
            ) where T
            amount = length(d_range)
            if length(s_range) != amount
                throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
            end
            amount == 0 && return dest
            d_offset = first(d_range)[1]
            s_offset = first(s_range)[1]
            copy!(dest, d_offset, src, s_offset, amount)
        end
        # function copy!{T, N}(
        #         dest::$D{T, N}, rdest::CartesianRange{CartesianIndex{N}},
        #         src::$S{T, N}, ssrc::CartesianRange{CartesianIndex{N}},
        #     )
        #     copy!(unpack_buffer(dest), rdest, unpack_buffer(src), ssrc)
        # end
        function copy!(
                dest::$D{T, N}, src::$S{T, N}
            ) where {T, N}
            len = length(src)
            len == 0 && return dest
            if length(dest) > len
                throw(BoundsError(dest, length(src)))
            end
            copy!(dest, 1, src, 1, len)
        end
    end
end


function copy_kernel!(state, dest, dest_offsets, src, src_offsets, shape, shape_dest, shape_source, length)
    i = linear_index(dest, state)
    if i <= length
        # TODO can this be done faster and smarter?
        idx = gpu_ind2sub(shape, i)
        dest_idx = gpu_sub2ind(shape_dest, idx .+ dest_offsets)
        src_idx = gpu_sub2ind(shape_source, idx .+ src_offsets)
        @inbounds dest[dest_idx] = src[src_idx]
    end
    return
end

function copy!(
        dest::AbstractAccArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::AbstractAccArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    ) where {T, N}
    shape = size(destcrange)
    if shape != size(srccrange)
        throw(DimensionMismatch("Ranges don't match their size. Found: $shape, $(size(srccrange))"))
    end
    len = length(destcrange)
    dest_offsets = Cuint.(destcrange.start.I .- 1)
    src_offsets = Cuint.(srccrange.start.I .- 1)
    ui_shape = Cuint.(shape)
    gpu_call(
        copy_kernel!, dest,
        (dest, dest_offsets, src, src_offsets, ui_shape, Cuint.(size(dest)), Cuint.(size(src)), Cuint(len)),
        len
    )
    dest
end


function copy!(
        dest::AbstractAccArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::AbstractArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    ) where {T, N}
    # Is this efficient? Maybe!
    # TODO: compare to a pure intrinsic copy implementation!
    # this would mean looping over linear sections of memory and
    # use copy!(dest, offset::Integer, buffer(src), offset::Integer, amout::Integer)
    src_gpu = typeof(dest)(map(idx-> src[idx], srccrange))
    nrange = CartesianRange(one(CartesianIndex{N}), CartesianIndex(size(src_gpu)))
    copy!(dest, destcrange, src_gpu, nrange)
    dest
end


function copy!(
        dest::AbstractArray{T, N}, destcrange::CartesianRange{CartesianIndex{N}},
        src::AbstractAccArray{T, N}, srccrange::CartesianRange{CartesianIndex{N}}
    ) where {T, N}
    # Is this efficient? Maybe!
    dest_gpu = similar(src, size(destcrange))
    nrange = CartesianRange(one(CartesianIndex{N}), CartesianIndex(size(dest_gpu)))
    copy!(dest_gpu, nrange, src, srccrange)
    copy!(dest, destcrange, Array(dest_gpu), nrange)
    dest
end


Base.copy(x::AbstractAccArray) = identity.(x)

indexlength(A, i, array::AbstractArray) = length(array)
indexlength(A, i, array::Number) = 1
indexlength(A, i, array::Colon) = size(A, i)

function Base.setindex!(A::AbstractAccArray{T, N}, value, indexes...) where {T, N}
    # similarly, value should always be a julia array
    shape = ntuple(Val{N}) do i
        indexlength(A, i, indexes[i])
    end
    if !isa(value, T) # TODO, shape check errors for x[1:3] = 1
        Base.setindex_shape_check(value, indexes...)
    end
    checkbounds(A, indexes...)
    v = array_convert(Array{T, N}, value)
    # since you shouldn't update GPUArrays with single indices, we simplify the interface
    # by always mapping to ranges
    ranges_dest = to_cartesian(A, indexes)
    ranges_src = CartesianRange(size(v))

    copy!(A, ranges_dest, v, ranges_src)
    return
end

function Base.getindex(A::AbstractAccArray{T, N}, indexes...) where {T, N}
    cindexes = Base.to_indices(A, indexes)
    # similarly, value should always be a julia array
    # We shouldn't really bother about checkbounds performance, since setindex/getindex will always be relatively slow
    checkbounds(A, cindexes...)

    shape = map(length, cindexes)
    result = Array{T, N}(shape)
    ranges_src = to_cartesian(A, cindexes)
    ranges_dest = CartesianRange(shape)
    copy!(result, ranges_dest, A, ranges_src)
    if all(i-> isa(i, Integer), cindexes) # scalar
        return result[]
    else
        return result
    end
end


#Broadcast
Base.@propagate_inbounds broadcast_index(::Val{false}, arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index(
        ::Val{true}, arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    ) where {T, N}
    @inbounds return arg[i]
end
@generated function broadcast_index(::Val{true}, arg::AbstractArray{T, N}, shape, i) where {T, N}
    idx = []
    for i = 1:N
        push!(idx, :(s[$i] < shape[$i] ? 1 : idx[$i]))
    end
    expr = quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        s = size(arg)
        idx = ind2sub(shape, i)
        @inbounds return arg[$(idx...)]
    end
end
Base.@propagate_inbounds broadcast_index(arg, shape, i) = arg
Base.@propagate_inbounds function broadcast_index(
        arg::AbstractArray{T, N}, shape::NTuple{N, Integer}, i
    ) where {T, N}
    return arg[i]
end
Base.@propagate_inbounds function broadcast_index(
        arg::AbstractArray, shape::Integer, i::Integer
    )
    return arg[i]
end
@generated function broadcast_index(arg::AbstractArray{T, N}, shape, i) where {T, N}
    idx = []
    for i = 1:N
        push!(idx, :(ifelse(s[$i] < shape[$i], 1, idx[$i])))
    end
    expr = quote
        $(Expr(:meta, :inline, :propagate_inbounds))
        s = size(arg)
        idx = ind2sub(shape, i)
        @inbounds return arg[$(idx...)]
    end
end


#=
reinterpret taken from julia base/array.jl
Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors:

https://github.com/JuliaLang/julia/contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#
import Base.reinterpret

"""
Unsafe reinterpret for backends to overload.
This makes it easier to do checks just on the high level.
"""
function unsafe_reinterpret end

function reinterpret(::Type{T}, a::AbstractAccArray{S,1}) where T where S
    nel = Int(div(length(a)*sizeof(S),sizeof(T)))
    # TODO: maybe check that remainder is zero?
    return reinterpret(T, a, (nel,))
end

function reinterpret(::Type{T}, a::AbstractAccArray{S}) where T where S
    if sizeof(S) != sizeof(T)
        throw(ArgumentError("result shape not specified"))
    end
    reinterpret(T, a, size(a))
end

function reinterpret(::Type{T}, a::AbstractAccArray{S}, dims::NTuple{N,Int}) where T where S where N
    if !isbits(T)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(T) is not a bits type"))
    end
    if !isbits(S)
        throw(ArgumentError("cannot reinterpret Array{$(S)} to ::Type{Array{$(T)}}, type $(S) is not a bits type"))
    end
    nel = div(length(a)*sizeof(S),sizeof(T))
    if prod(dims) != nel
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(nel)"))
    end
    unsafe_reinterpret(T, a, dims)
end

function Base.reshape(a::AbstractAccArray{T}, dims::NTuple{N,Int}) where T where N
    if prod(dims) != length(a)
        throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(length(a))"))
    end
    unsafe_reinterpret(T, a, dims)
end
