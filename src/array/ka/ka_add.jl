import KernelAbstractions

@kernel function matrix_add_kernel!(C, alpha, A, beta, B, M, N)


    i, j = @index(Local, NTuple)
    gi, gj = @index(Global, NTuple)

    if gi <= M && gj <= N
        @inbounds C[gi, gj] = alpha * A[gi, gj] + beta * B[gi, gj]
    end

end

function matrix_add!(C, alpha, A, beta, B; nthreads = (32, 32))
    if typeof(A) != typeof(B) != typeof(C)
        error("Types of A, B, and C are different!")
    end

    if length(A) != length(B) != length(C)
        error("Lengths of A, B, and C are different!")
    end
    M, N = size(A)
    backend = get_backend(A)

    kernel = matrix_add_kernel!(backend, nthreads)
    kernel(C, alpha, A, beta, B, M, N; ndrange = size(A))
    KernelAbstractions.synchronize(backend)
end
