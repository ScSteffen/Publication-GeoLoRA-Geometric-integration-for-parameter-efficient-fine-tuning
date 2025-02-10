#%%
            
import torch
import peft
from layer import AdaLoraLayer
from config import AdaLoraConfig
from model import AdaLoraModel
from extract_weights import collect_lr_layers

class autoenc(torch.nn.Module):
    def __init__(self):
        super(autoenc,self).__init__()
        self.seq1 = torch.nn.Linear(100,40)
        self.seq2 = torch.nn.Linear(40,100)
        self.nl = torch.nn.Tanh()
    def forward(self,x):
        return self.seq2(self.nl(self.seq1(x)))
  
model = autoenc()
model.train()  
for p in model.parameters():
    p.requires_grad = False

adapter_name = "dlrt"
peft_config = AdaLoraConfig(inference_mode=False, init_r=1, lora_alpha=32, lora_dropout=0.0,target_modules=['seq1','seq2'],bias = 'all'
)

model = AdaLoraModel(model,peft_config,adapter_name)
# model = get_peft_model(model,peft_config)

adapter_layers = collect_lr_layers(model,AdaLoraLayer)   #### collect the lora DLRT layers

# optimizer = rlora_AdamW(get_param_groups_rlora(model),lr = 0.001,weight_decay=0.0001,tau = 0.1,frequency_qr=1,r_grad=False)
optimizer = torch.optim.AdamW([l.bias for l in adapter_layers],weight_decay=0.0)
criterion = torch.nn.MSELoss()
X = torch.randn((80,100))
for _ in range(100):
    optimizer.zero_grad()
    y_hat = model(X)
    loss = criterion(y_hat,X)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad and 'lora_B' in n:
                print(f'rank {n}: {torch.linalg.matrix_rank(p)}')
    print(loss.item())

print(f'DLRT step')
for l in adapter_layers:
    print(f'params before step: {l.lora_E[adapter_name]}')
    l.step(1e-2,1e-1,'dlrt')
    print(f'params after step: {l.lora_E[adapter_name]}')
# %%
