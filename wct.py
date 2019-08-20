import torch

def mean_covsqrt(f, inverse=False, eps=1e-5):
    c, h, w = f.size()
    
    f_mean = torch.mean(f.view(c, h*w), dim=1, keepdim=True)
    f_zeromean = f.view(c, h*w) - f_mean
    f_cov = torch.mm(f_zeromean, f_zeromean.t())

    u, s, v = torch.svd(f_cov)

    k = c
    for i in range(c):
        if s[i] < eps:
            k = i
            break
            
    if inverse:
        p = -0.5
    else:
        p = 0.5
        
    f_covsqrt = torch.mm(torch.mm(v[:, 0:k], torch.diag(s[0:k].pow(p))), v[:, 0:k].t())
    return f_mean, f_covsqrt

def whitening(f):
    c, h, w = f.size()

    f_mean, f_inv_covsqrt = mean_covsqrt(f, inverse=True)
    
    whiten_f = torch.mm(f_inv_covsqrt, f.view(c, h*w) - f_mean)
    
    return whiten_f.view(c, h, w)

def coloring(f, t):
    f_c, f_h, f_w = f.size()
    t_c, t_h, t_w = t.size()
    
    t_mean, t_covsqrt = mean_covsqrt(t)
    
    colored_f = torch.mm(t_covsqrt, f.view(f_c, f_h*f_w)) + t_mean
    
    return colored_f.view(f_c, f_h, f_w)

def batch_whitening(f):
    b, c, h, w = f.size()

    whiten_f = torch.Tensor(b, c, h, w).type_as(f)
    for i, f_ in enumerate(torch.split(f, 1)):
        whiten_f[i] = whitening(f_.squeeze())
        
    return whiten_f

def batch_coloring(f, t):
    b, c, h, w = f.size()

    colored_f = torch.Tensor(b, c, h, w).type_as(f)
    for i, (f_, t_) in enumerate(zip(torch.split(f, 1), torch.split(t, 1))):
        colored_f[i] = coloring(f_.squeeze(), t_.squeeze())

    return colored_f
