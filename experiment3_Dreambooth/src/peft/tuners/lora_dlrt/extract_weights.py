import torch,math


@torch.no_grad()
def get_param_groups_rlora(model):
    '''
    extract weights in the correct format to pass to the optimizer
    '''
    lora_A = []
    lora_B = []
    lora_E = []
    biases = []

    for n,p in model.named_parameters():
        if p.requires_grad:
            if 'lora_A' in n:
                lora_A.append(p)
            elif 'lora_B' in n:
                lora_B.append(p)
            elif 'lora_E' in n:
                lora_E.append(p)
            elif 'bias' in n:
                biases.append(p)
    return [{'params':lora_A},{'params':lora_B},{'params':lora_E},{'params':biases}]


@torch.no_grad()
def collect_lr_layers(model,instance_module):
    '''
    Collect all layers that have adapters of the type instance_module in a list
    '''
    adapters_layers = []
    def fn(mod):
        if isinstance(mod,instance_module):
            adapters_layers.append(mod)
    model.apply(fn)
    return adapters_layers

@torch.no_grad()
def count_params_conv_lora(convs_list):
    total_params = 0
    for l in convs_list:
        total_params+= int(torch.prod(torch.tensor(l.lora_A['default'].weight.shape)) + torch.prod(torch.tensor(l.lora_B['default'].weight.shape)))
    print(f'total params conv lora {total_params}')

@torch.no_grad()
def count_params_conv_tdlrt(convs_list):
    total_params = 0
    for l in convs_list:
        total_params+= math.prod(l.rank) + sum([r*d for (r,d) in zip(l.rank,l.dims)])
    print(f'total params conv tdlora {total_params}')


@torch.no_grad()
def count_params_conv_pdlrt(layers_list):
    total_params = 0
    for l in layers_list:
        total_params+= l.r*l.r + l.r*l.in_features +l.r*l.out_features
    print(f'total params conv tdlora {total_params}')


@torch.no_grad()
def count_params_conv_adalora(convs_list):
    total_params = 0
    for l in convs_list:
        total_params+= int(torch.prod(torch.tensor(l.lora_A['default'].weight[:l.r,:].shape)) + torch.prod(torch.tensor(l.lora_B['default'].weight[:,:l.r].shape))) + torch.prod(torch.tensor(l.lora_E['default'].weight[:l.r,:].shape))
    print(f'total params conv lora {total_params}')