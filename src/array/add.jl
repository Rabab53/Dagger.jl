function outplace_add!(
    C::DMatrix{T},
    transA::Char,
    transB::Char,
    A::DMatrix{T},
    B::DMatrix{T},
    _add::LinearAlgebra.MulAddMul,
) where {T}
    partC, partA, partB = _repartition_add(C, A, B, transA, transB)
    return maybe_copy_buffered(C=>partC, A=>partA, B=>partB) do C, A, B
        return outplace_add_dagger!(C, transA, transB, A, B, _add)
    end
end

function inplace_add!(
    transA::Char,
    transB::Char,
    A::DMatrix{T},
    B::DMatrix{T},
    _add::LinearAlgebra.MulAddMul,
) where {T}
    partA, partB = _repartition_add(C, A, B, transA, transB)
    return maybe_copy_buffered(A=>partA, B=>partB) do A, B
        return inplace_add_dagger!(B, transA, transB, A, B, _add)
    end
end

function _repartition_add(C, A, B, transA::Char, transB::Char)
    partA = A.partitioning.blocksize
    partB = B.partitioning.blocksize
    istransA = transA == 'T' || transA == 'C'
    istransB = transB == 'T' || transB == 'C'
    dimA = !istransA ? partA[1] : partA[2]
    dimB = !istransB ? partB[2] : partB[1]
    dimA_other = !istransA ? partA[2] : partA[1]
    dimB_other = !istransB ? partB[1] : partB[2]

    # If A and B rows/cols don't match, fix them
    # Uses the smallest blocking of all dimensions
    sz = minimum((partA[1], partA[2], partB[1], partB[2]))
    if dimA != dimB
        dimA = dimB = sz
        if !istransA
            partA = (sz, partA[2])
        else
            partA = (partA[1], sz)
        end
        if !istransB
            partB = (partB[1], sz)
        else
            partB = (sz, partB[2])
        end
    end
    if dimA_other != dimB_other
        dimA_other = dimB_other = sz
        if !istransA
            partA = (partA[1], sz)
        else
            partA = (sz, partA[2])
        end
        if !istransB
            partB = (sz, partB[2])
        else
            partB = (partB[1], sz)
        end
    end

    if A === B && ((!istransA && istransB) || (istransA && !istransB))
        # syrk requires A to be square blocks
        partA = (sz, sz)
        dimA = dimB = sz
    end

    # Ensure C partitioning matches A * B
    partC = (dimA, dimB)

    return Blocks(partC...), Blocks(partA...), Blocks(partB...)
end

function _repartition_add(A, B, transA::Char, transB::Char)
    partA = A.partitioning.blocksize
    partB = B.partitioning.blocksize
    istransA = transA == 'T' || transA == 'C'
    istransB = transB == 'T' || transB == 'C'
    dimA = !istransA ? partA[1] : partA[2]
    dimB = !istransB ? partB[2] : partB[1]
    dimA_other = !istransA ? partA[2] : partA[1]
    dimB_other = !istransB ? partB[1] : partB[2]

    # If A and B rows/cols don't match, fix them
    # Uses the smallest blocking of all dimensions
    sz = minimum((partA[1], partA[2], partB[1], partB[2]))
    if dimA != dimB
        dimA = dimB = sz
        if !istransA
            partA = (sz, partA[2])
        else
            partA = (partA[1], sz)
        end
        if !istransB
            partB = (partB[1], sz)
        else
            partB = (sz, partB[2])
        end
    end
    if dimA_other != dimB_other
        dimA_other = dimB_other = sz
        if !istransA
            partA = (partA[1], sz)
        else
            partA = (sz, partA[2])
        end
        if !istransB
            partB = (sz, partB[2])
        else
            partB = (partB[1], sz)
        end
    end

    if A === B && ((!istransA && istransB) || (istransA && !istransB))
        # syrk requires A to be square blocks
        partA = (sz, sz)
        dimA = dimB = sz
    end

    # Ensure C partitioning matches A * B
    partC = (dimA, dimB)

    return Blocks(partA...), Blocks(partB...)
end

function _outplace_add!(transA::Char, transB::Char, alpha::T, A, beta::T, B, C) where T
    Am, An = size(A)
    Bm, Bn = size(B)
    Cm, Cn = size(C)
    if transA == 'N' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            C[i,j] = alpha * A[i,j] + beta * B[i,j]
        end
    elseif transA == 'N' && transB == 'T'
        for i in 1:Cm, j in 1:Cn
            C[i,j] = alpha * A[i,j] + beta * B[j,i]
        end
    elseif transA == 'N' && transB == 'C'
        for i in 1:Cm, j in 1:Cn
            C[i,j] = alpha * A[i,j] + beta * conj(B[i,j])
        end
    elseif transA == 'T' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            C[i,j] = alpha * A[j,i] + beta * B[i,j]
        end
    else if transA == 'C' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            C[i,j] = alpha * conj(A[j,i]) + beta * B[i,j]
        end
    elseif transA == 'T' && transB == 'T'
        for i in 1:Cm, j in 1:Cn
            C[j,i] = alpha * A[j,i] + beta * B[j,i]
        end
    elseif transA == 'C' && transB == 'C'
        for i in 1:Cm, j in 1:Cn
            C[j,i] = alpha * conj(A[j,i]) + beta * conj(B[j,i])
        end
    end

end
function _inplace_add!(transA::Char, transB::Char, alpha::T, A, beta::T, B) where T
    Am, An = size(A)
    Bm, Bn = size(B)
    if transA == 'N' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            B[i,j] = alpha * A[i,j] + beta * B[i,j]
        end
    elseif transA == 'N' && transB == 'T'
        for i in 1:Cm, j in 1:Cn
            B[j,i] = alpha * A[i,j] + beta * B[j,i]
        end
    elseif transA == 'N' && transB == 'C'
        for i in 1:Cm, j in 1:Cn
            B[j,i] = alpha * A[i,j] + beta * conj(B[j,i])
        end
    elseif transA == 'T' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            B[i,j] = alpha * A[j,i] + beta * B[i,j]
        end
    else if transA == 'C' && transB == 'N'
        for i in 1:Cm, j in 1:Cn
            B[i,j] = alpha * conj(A[j,i]) + beta * B[i,j]
        end
    elseif transA == 'T' && transB == 'T'
        for i in 1:Cm, j in 1:Cn
            B[j,i] = alpha * A[j,i] + beta * B[j,i]
        end
    elseif transA == 'C' && transB == 'C'
        for i in 1:Cm, j in 1:Cn
            B[j,i] = alpha * conj(A[j,i]) + beta * conj(B[j,i])
        end
    end
end
"""
Performs  matrix-matrix addition or subtraction

C = alpha op( A ) + beta op( B )

where op( X ) is one of

op( X ) = X  or op( X ) = X' or op( X ) = g( X' )

alpha and beta are scalars, and A, B and C  are matrices, with op( A )
an m by n matrix, op( B ) a m by n matrix and C an m by n matrix.
"""
function outplace_add_dagger!(
    C::DMatrix{T},
    transA::Char,
    transB::Char,
    A::DMatrix{T},
    B::DMatrix{T},
    _add::LinearAlgebra.MulAddMul,
) where {T}
    Ac = A.chunks
    Bc = B.chunks
    Cc = C.chunks
    Amt, Ant = size(Ac)
    Bmt, Bnt = size(Bc)
    Cmt, Cnt = size(Cc)

    alpha = _add.alpha
    beta = _add.beta

    if Ant != Bmt
        throw(DimensionMismatch(lazy"A has number of blocks ($Amt,$Ant) but B has number of blocks ($Bmt,$Bnt)"))
    end

    Dagger.spawn_datadeps() do
        for m in range(1, Cmt)
            for n in range(1, Cnt)
                if transA == 'N'
                    if transB == 'N'
                        # A: NoTrans / B: NoTrans
                        Dagger.@spawn _outplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[m, n]),
                            beta,
                            In(Bc[m, n]),
                            InOut(Cc[m, n]),
                        )
                    else
                        # A: NoTrans / B: [Conj]Trans
                        Dagger.@spawn _outplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[m, n]),
                            beta,
                            In(Bc[n, m]),
                            InOut(Cc[m, n]),
                        )
                    end
                else
                    if transB == 'N'
                        # A: [Conj]Trans / B: NoTrans
                        Dagger.@spawn _outplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[n, m]),
                            beta,
                            In(Bc[m, n]),
                            InOut(Cc[m, n]),
                        )
                    else
                        # A: [Conj]Trans / B: [Conj]Trans
                        Dagger.@spawn _outplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[n, m]),
                            beta,
                            In(Bc[n, m]),
                            InOut(Cc[n, m]),
                        )
                    end
                end
            end
        end
    end

    return C
end

function inplace_add_dagger!(
    transA::Char,
    transB::Char,
    A::DMatrix{T},
    B::DMatrix{T},
    _add::LinearAlgebra.MulAddMul,
) where {T}
    Ac = A.chunks
    Bc = B.chunks
    Amt, Ant = size(Ac)
    Bmt, Bnt = size(Bc)

    alpha = _add.alpha
    beta = _add.beta

    if Ant != Bmt
        throw(DimensionMismatch(lazy"A has number of blocks ($Amt,$Ant) but B has number of blocks ($Bmt,$Bnt)"))
    end

    Dagger.spawn_datadeps() do
        for m in range(1, Cmt)
            for n in range(1, Cnt)
                if transA == 'N'
                    if transB == 'N'
                        # A: NoTrans / B: NoTrans
                        Dagger.@spawn _inplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[m, n]),
                            beta,
                            InOut(Bc[m, n]),
                        )
                    else
                        # A: NoTrans / B: [Conj]Trans
                        Dagger.@spawn _inplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[m, n]),
                            beta,
                            InOut(Bc[n, m]),
                        )
                    end
                else
                    if transB == 'N'
                        # A: [Conj]Trans / B: NoTrans
                        Dagger.@spawn _inplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[n, m]),
                            beta,
                            InOut(Bc[m, n]),
                        )
                    else
                        # A: [Conj]Trans / B: [Conj]Trans
                        Dagger.@spawn _inplace_add!(
                            transA,
                            transB,
                            alpha,
                            In(Ac[n, m]),
                            beta,
                            InOut(Bc[n, m]),
                        )
                    end
                end
            end
        end
    end
    return B
end
