import torch
from torch.autograd import grad
from tqdm import tqdm


def apply_perturb(model, v):
    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr:curr+length].data
        curr += length


def sam_grad(model, loss):
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params, allow_unused=True)
    sample_grad = [(torch.zeros_like(param) if grad is None else grad).view(-1)
                   for grad, param in zip(sample_grad, model.parameters())]
    return torch.cat(sample_grad)


def apply_perturb(model, v):
    curr = 0
    with torch.no_grad():
        for param in model.parameters():
            length = param.view(-1).shape[0]
            param += v[curr:curr+length].view(param.shape)
            curr += length


def woodfisher(model, train_dl, criterion, v):
    model.eval()
    k_vec = torch.clone(v)
    N = 1000# len(train_dl)

    for idx, (inp, mask, label) in enumerate(tqdm(train_dl)):
        if idx == N:
            break
        inp = inp.cuda()
        mask = mask.cuda()
        label = label.cuda()

        model.zero_grad()

        output = model(inp, mask)
        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if idx == 0:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec

    return k_vec


def Wfisher(data_loaders, model, criterion, args):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    retain_grad_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=128, shuffle=False)
    retain_loader = torch.utils.data.DataLoader(
        retain_loader.dataset, batch_size=1, shuffle=False)
    forget_loader = torch.utils.data.DataLoader(
        forget_loader.dataset, batch_size=128, shuffle=False)
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).cuda()
    retain_grad = torch.zeros_like(torch.cat(params)).cuda()
    total = 0
    model.eval()
    for i, (inp, mask, label) in enumerate(tqdm(forget_loader)):
        inp = inp.cuda()
        mask = mask.cuda()
        label = label.cuda()

        model.zero_grad()

        real_num = inp.shape[0]
        output = model(inp, mask)
        loss = criterion(output, label)
        f_grad = sam_grad(model, loss)*real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    for i, (inp, mask, label) in enumerate(tqdm(retain_grad_loader)):
        inp = inp.cuda()
        mask = mask.cuda()
        label = label.cuda()

        model.zero_grad()

        real_num = inp.shape[0]
        output = model(inp, mask)
        loss = criterion(output, label)
        r_grad = sam_grad(model, loss)*real_num
        retain_grad += r_grad
        total_2 += real_num

    retain_grad *= (total/((total+total_2)*total_2))
    forget_grad /= total+total_2
    perturb = woodfisher(model, retain_loader,
                         criterion=criterion, v=forget_grad-retain_grad)

    apply_perturb(model, args.alpha*perturb)

    return model
