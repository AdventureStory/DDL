import torch
from functools import partial

def guassian_kernel(source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
        source: 源域数据（n * len(x))
        target: 目标域数据（m * len(y))
        kernel_mul:
        kernel_num: 取不同高斯核的数量
        fix_sigma: 不同高斯核的sigma值
    Return:
        sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0) #将source和target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)

def get_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        计算源域数据和目标域数据的MMD距离
        Params:
            source: 源域数据（n * len(x))
            target: 目标域数据（m * len(y))
            kernel_mul:
            kernel_num: 取不同高斯核的数量
            fix_sigma: 不同高斯核的sigma值
        Return:
            loss: MMD loss
        '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分为4份
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1).float()
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = torch.FloatTensor(sigmas)
        )

    
    '''
    if params.use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = torch.cuda.FloatTensor(sigmas)
        )
    else:
        
    '''
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value

def test2():
    source = torch.tensor([[1,2,3]]) # <class 'torch.Tensor'>
    target = torch.tensor([[1,1,2]])
    result = mmd_loss(source,target)
    print(result) # tensor(6.1863) <class 'torch.Tensor'>

    
    source = torch.randn([32, 10, 1, 1])
    source =torch.squeeze(source,3)
    source =torch.squeeze(source,2)
    target = torch.randn([32, 10, 1, 1]) # 第一维度可以不相同，但第二个维度必须相同，否则会出错。
    print(source.size())
    print(target.size())
    target =torch.squeeze(target,3)
    target =torch.squeeze(target,2)
    print(source.size())
    print(target.size())
    result = mmd_loss(source,target)
    print(result) # tensor(2.0059)
    


def test():
    # 注意source和target须为tensor，最终的MMD损失也为tensor。
    # 另外，source和target须为一个batch，如果源域和目标域数据为单个样本的话，则需要增加一个维度。
    # 对了，source和target的样本个数不要求一样，但第二维度须一样。
    source = torch.tensor([[1,2,3]]) # <class 'torch.Tensor'>
    target = torch.tensor([[1,1,2]])
    result = get_MMD(source,target)
    print(result) # tensor(6.1863) <class 'torch.Tensor'>

    source = torch.tensor([[2,3,66]])
    target = torch.tensor([[1,5,6]])
    result = get_MMD(source,target)
    print(result) # tensor(6.1863)
    # 不知道为什么这组MMD损失和上面那组结果一样。

    source = torch.tensor([[1,2,3],[2,3,66]])
    target = torch.tensor([[1,1,2],[1,5,6]])
    result = get_MMD(source,target)
    print(result) # tensor(1.8873)
    # 这组MMD损失主要是为了检验一下，MMD损失是不是(上述两组)单个样本对应的MMD损失的和。
    # 显然，结果表明不是的。MMD损失计算的是一批源域数据和对应目标域那批数据的分布损失，是从整体考虑的分布损失，不是简单地理解为逐样本MMD损失的和。

    source = torch.tensor([[1,2,3],[2,3,66]])
    target = torch.tensor([[1,2,2],[1,5,60]])
    result = get_MMD(source,target)
    print(result) # tensor(0.0621)
    # 这组MMD损失主要是为了检验一下，是不是样本个数和维度相同，MMD损失就一样(因为前两组MMD损失一模一样)。
    # 结果显然不是的。所以，为什么前两组MMD损失一样呢？有没有同学可以解释一下？

    source = torch.tensor([[1,2,3]])
    target = torch.tensor([[1,2,2],[2,3,6]]) # 第一维度可以不相同，但第二个维度必须相同，否则会出错。
    print(source.size())
    print(target.size())
    result = get_MMD(source,target)
    print(result) # tensor(2.0059)


    source = torch.randn([32, 10, 1, 1])
    source =torch.squeeze(source,3)
    source =torch.squeeze(source,2)
    target = torch.randn([32, 10, 1, 1]) # 第一维度可以不相同，但第二个维度必须相同，否则会出错。
    print(source.size())
    print(target.size())
    target =torch.squeeze(target,3)
    target =torch.squeeze(target,2)
    print(source.size())
    print(target.size())
    result = get_MMD(source,target)
    print(result) # tensor(2.0059)

test2()

