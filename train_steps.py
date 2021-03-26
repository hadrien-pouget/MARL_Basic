import torch

from utils import zero_grad

def naive_step(p1, p2, v1, v2, lr):
    zero_grad(p1)
    zero_grad(p2)

    ### Gradients of each value function w/ respect to each set of parameters
    [p1_grad] = torch.autograd.grad(v1, [p1], create_graph=True, only_inputs=True)
    [p2_grad] = torch.autograd.grad(v2, [p2], create_graph=True, only_inputs=True)

    ### Take gradient steps
    with torch.no_grad():
        p1 += p1_grad * lr
        p2 += p2_grad * lr
    
    return p1, p2

def lola_step(p1, p2, v1, v2, lr):
    zero_grad(p1)
    zero_grad(p2)

    ### Gradients of each value function w/ respect to each set of parameters
    [v1_grad_p1, v1_grad_p2] = torch.autograd.grad(v1, [p1, p2], create_graph=True, only_inputs=True)
    [v2_grad_p1, v2_grad_p2] = torch.autograd.grad(v2, [p1, p2], create_graph=True, only_inputs=True)

    v1_grad_p1 = v1_grad_p1.reshape(-1)
    v1_grad_p2 = v1_grad_p2.reshape(-1)
    v2_grad_p1 = v2_grad_p1.reshape(-1)
    v2_grad_p2 = v2_grad_p2.reshape(-1)

    ### p1 lola gradient
    multiply = torch.dot(v2_grad_p2, v1_grad_p2)
    v1_approx = v1 + multiply
    [p1_grad] = torch.autograd.grad(v1_approx, p1, retain_graph=True, only_inputs=True)

    ### p2 lola gradient
    multiply = torch.dot(v1_grad_p1, v2_grad_p1)
    v2_approx = v2 + multiply
    [p2_grad] = torch.autograd.grad(v2_approx, p2, retain_graph=True, only_inputs=True)

    ### Take gradient steps
    with torch.no_grad():
        p1 += p1_grad * lr
        p2 += p2_grad * lr

    return p1, p2
