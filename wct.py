import torch

def covsqrt_mean(feature, inverse=False, tolerance=1e-14):
    # I referenced the default svd tolerance value in matlab.

    b, c, h, w = feature.size()

    mean = torch.mean(feature.view(b, c, -1), dim=2, keepdim=True)
    zeromean = feature.view(b, c, -1) - mean
    cov = torch.bmm(zeromean, zeromean.transpose(1, 2))

    evals, evects = torch.symeig(cov, eigenvectors=True)
    
    p = 0.5
    if inverse:
        p *= -1

    covsqrt = []
    for i in range(b):
        k = 0
        for j in range(c):
            if evals[i][j] > tolerance:
                k = j
                break
        covsqrt.append(torch.mm(evects[i][:, k:],
                            torch.mm(evals[i][k:].pow(p).diag_embed(),
                                     evects[i][:, k:].t())).unsqueeze(0))
    covsqrt = torch.cat(covsqrt, dim=0)

    return covsqrt, mean
    

def whitening(feature):
    b, c, h, w = feature.size()
    
    inv_covsqrt, mean = covsqrt_mean(feature, inverse=True)

    normalized_feature = torch.matmul(inv_covsqrt, feature.view(b, c, -1)-mean)
    
    return normalized_feature.view(b, c, h, w)


def coloring(feature, target):
    b, c, h, w = feature.size()

    covsqrt, mean = covsqrt_mean(target)
    
    colored_feature = torch.matmul(covsqrt, feature.view(b, c, -1)) + mean
    
    return colored_feature.view(b, c, h, w)
