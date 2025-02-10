import torch 


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
