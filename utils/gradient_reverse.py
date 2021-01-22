from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    """
    The return x.view_as(x) seems to be necessary, because otherwise backward is not being called,
    I guess that as optimization Autograd checks if the Function modified the tensor to see if backward should be called.
    
    From https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3
    """

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class NeutralizeLayerF(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * 0.0
        return output, None
