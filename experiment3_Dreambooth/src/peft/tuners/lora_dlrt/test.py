#%%
            
import torch
from peft.tuners.lora_dlrt.layer import LoraLayer
from peft import DLRT_config
from peft import DLRT_model
from peft.tuners.lora_dlrt.extract_weights import collect_lr_layers

class autoenc(torch.nn.Module):
    def __init__(self):
        super(autoenc,self).__init__()
        self.conv1 = torch.nn.Conv2d(3,4,3)
        self.flatten = torch.nn.Flatten()
        self.seq1 = torch.nn.Linear(400,100)
        self.nl = torch.nn.Tanh()
    def forward(self,x):
        return self.seq1(self.flatten(self.nl(self.conv1(x))))
  
model = autoenc()
model.eval()
model.train()  
for p in model.parameters():
    p.requires_grad = False

adapter_name = "dlrt"
peft_config = DLRT_config(inference_mode=False, init_r=1, lora_alpha=32, lora_dropout=0.0,target_modules=['seq1','conv1'],bias = 'all'
)

model = DLRT_model(model,peft_config,adapter_name)
# model = get_peft_model(model,peft_config)

adapter_layers = collect_lr_layers(model,LoraLayer)   #### collect the lora DLRT layers
print(adapter_layers)
print([n for l in adapter_layers for n,p in l.named_parameters()])
input()

# optimizer = rlora_AdamW(get_param_groups_rlora(model),lr = 0.001,weight_decay=0.0001,tau = 0.1,frequency_qr=1,r_grad=False)
optimizer = torch.optim.AdamW([l.bias for l in adapter_layers],weight_decay=0.0)
criterion = torch.nn.MSELoss()
X = torch.randn((80,3,12,12))
Y = torch.randn((80,100))
for _ in range(100):
    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat,Y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad and 'lora_B' in n:
                print(f'rank {n}: {torch.linalg.matrix_rank(p)}')
    print(loss.item())

print(f'DLRT step')
# for l in adapter_layers:
#     print(f'params before step: {l.lora_E[adapter_name]}')
#     l.step(1e-2,1e-1,'dlrt')
#     print(f'params after step: {l.lora_E[adapter_name]}')
# %%
