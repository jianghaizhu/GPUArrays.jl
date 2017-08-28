
is_struct(::Type{T}) where {T} = !(sizeof(T) != 0 && nfields(T) == 0)
is_glsl_primitive(::Type{T}) where {T <: StaticVector} = true
is_glsl_primitive(::Type{T}) where {T <: Union{Float32, Int32}} = true
is_glsl_primitive(T) = false

const max_batch_size = 1024

"""
Statically sized uniform buffer.
Supports push!, but with fixed memory, so it will error after reaching
it's preallocated length.
"""
mutable struct UniformBuffer{T, N}
    buffer::GLBuffer{T}
    offsets::NTuple{N, Int}
    elementsize::Int
    length::Int
end
const GLSLScalarTypes = Union{Float32, Int32, UInt32}
Base.eltype(::UniformBuffer{T, N}) where {T, N} = T


function glsl_alignement_size(T)
    T <: Bool && return sizeof(Int32), sizeof(Int32)
    N = sizeof(T)
    T <: GLSLScalarTypes && return N, N
    T <: Function && return sizeof(Vec4f0), sizeof(Vec4f0) # sizeof(EmptyStruct) padded to Vec4f0
    ET = eltype(T)
    if T <: Mat4f0
        a, s = glsl_alignement_size(Vec4f0)
        return a, 4s
    end
    N = sizeof(ET)
    if T <: Vec2f0
        return 2N, 2N
    end
    if T <: Vec4f0
        return 4N, 4N
    end
    if T <: Vec3f0
        return 4N, 3N
    end
    error("Struct $T not supported yet. Please help by implementing all rules from https://khronos.org/registry/OpenGL/specs/gl/glspec45.core.pdf#page=159")
end

function std140_offsets(::Type{T}) where T
    elementsize = 0
    offsets = if T <: GLSLScalarTypes
        elementsize = sizeof(T)
        (0,)
    else
        offset = 0
        offsets = ntuple(nfields(T)) do i
            ft = fieldtype(T, i)
            alignement, sz = glsl_alignement_size(ft)
            if offset % alignement != 0
                offset = (div(offset, alignement) + 1) * alignement
            end
            of = offset
            offset += sz
            of
        end
        elementsize = offset
        offsets
    end
    offsets, elementsize
end

"""
    Pre allocates an empty buffer with `max_batch_size` size
    which can be used to store multiple uniform blocks of type T
"""
function UniformBuffer(::Type{T}, max_batch_size = 1024, mode = GL_STATIC_DRAW) where T
    offsets, elementsize = std140_offsets(T)
    buffer = GLBuffer{T}(
        max_batch_size,
        elementsize * max_batch_size,
        GL_UNIFORM_BUFFER, mode
    )
    UniformBuffer(buffer, offsets, elementsize, 0)
end

"""
    Creates an Uniform buffer with the contents of `data`
"""
function UniformBuffer(data::T, mode = GL_STATIC_DRAW) where T
    buffer = UniformBuffer(T, 1, mode)
    push!(buffer, data)
    buffer
end

function assert_blocksize(buffer::UniformBuffer, program, blockname::String)
    block_index = glGetUniformBlockIndex(program, blockname)
    blocksize_ref = Ref{GLint}(0)
    glGetActiveUniformBlockiv(
        program, block_index,
        GL_UNIFORM_BLOCK_DATA_SIZE, blocksize_ref
    )
    blocksize = blocksize_ref[]
    @assert buffer.elementsize * length(buffer.buffer) == blocksize
end

_getfield(x::GLSLScalarTypes, i) = x
_getfield(x, i) = getfield(x, i)

function iterate_fields(buffer::UniformBuffer{T, N}, x, index) where {T, N}
    offset = buffer.elementsize * (index - 1)
    x_ref = isimmutable(x) ? Ref(x) : x
    base_ptr = Ptr{UInt8}(pointer_from_objref(x_ref))
    ntuple(Val{N}) do i
        offset + buffer.offsets[i], base_ptr + fieldoffset(T, i), sizeof(fieldtype(T, i))
    end
end

function Base.setindex!(buffer::UniformBuffer{T, N}, element::T, idx::Integer) where {T, N}
    if idx > length(buffer.buffer)
        throw(BoundsError(buffer, idx))
    end
    buff = buffer.buffer
    glBindBuffer(buff.buffertype, buff.id)
    dptr = Ptr{UInt8}(glMapBuffer(buff.buffertype, GL_WRITE_ONLY))
    for (offset, ptr, size) in iterate_fields(buffer, element, idx)
        unsafe_copy!(dptr + offset, ptr, size)
    end
    glUnmapBuffer(buff.buffertype)
    GLAbstraction.bind(buff, 0)
    element
end



function Base.push!(buffer::UniformBuffer{T, N}, element::T) where {T, N}
    buffer.length += 1
    buffer[buffer.length] = element
    buffer
end

function check_copy_bounds(
        dest, d_offset::Integer,
        src, s_offset::Integer,
        amount::Integer
    )
    amount > 0 || throw(ArgumentError(string("tried to copy n=", amount, " elements, but amount should be nonnegative")))
    if s_offset < 1 || d_offset < 1 ||
            s_offset + amount - 1 > length(src) ||
            d_offset + amount - 1 > length(dest)
        throw(BoundsError())
    end
    nothing
end


function copy!(
        dest::gl.GLBuffer{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::Vector{T}, s_range::CartesianRange{CartesianIndex{1}},
    ) where T
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount
    bind(dest)
    glBufferSubData(dest.buffertype, multiplicator * (d_offset - 1), nsz, Ref(src, s_offset))
    bind(dest, 0)
end

function copy!(
        dest::Vector{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::gl.GLBuffer{T}, s_range::CartesianRange{CartesianIndex{1}},
    ) where T
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount
    bind(src)
    glGetBufferSubData(
        src.buffertype, multiplicator * (s_offset - 1), nsz,
        Ref(dest, d_offset)
    )
    bind(src, 0)
    dest
end


# copy between two buffers
function copy!(
        dest::gl.GLBuffer{T}, d_range::CartesianRange{CartesianIndex{1}},
        src::gl.GLBuffer{T}, s_range::CartesianRange{CartesianIndex{1}}
    ) where T
    amount = length(d_range)
    if length(s_range) != amount
        throw(ArgumentError("Copy range needs same length. Found: dest: $amount, src: $(length(s_range))"))
    end
    amount == 0 && return dest
    d_offset = first(d_range)[1]
    s_offset = first(s_range)[1]
    check_copy_bounds(dest, d_offset, src, s_offset, amount)
    multiplicator = sizeof(T)
    nsz = multiplicator * amount

    glBindBuffer(GL_COPY_READ_BUFFER, src.id)
    glBindBuffer(GL_COPY_WRITE_BUFFER, dest.id)
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
        multiplicator * (s_offset - 1),
        multiplicator * (d_offset - 1),
        multiplicator * amount
    )
    glBindBuffer(GL_COPY_READ_BUFFER, 0)
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0)
    return nothing
end
