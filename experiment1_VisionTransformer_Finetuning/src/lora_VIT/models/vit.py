# %%
import torch
from transformers import ViTForImageClassification
import sys, os
import peft

sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.lora_VIT.lora_dlrt.extract_weights import collect_lr_layers
from src.lora_VIT.lora_dlrt.config import AdaLoraConfig
from src.lora_VIT.lora_dlrt.model import AdaLoraModel
from src.lora_VIT.lora_dlrt.layer import AdaLoraLayer


def vit_b32_dlrt(r=16, adapter_name="dlrt", n_classes=10):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    peft_config = AdaLoraConfig(
        orth_reg_weight=0.0,
        inference_mode=False,
        init_r=r,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=[
            "query",
            "key",
            "value",
            # "classifier",
            "intermediate.dense",  # might throw error
            "output.dense",
        ],
        bias="all",
    )  #### just some layers

    model = AdaLoraModel(model, peft_config, adapter_name)
    adapter_layers = collect_lr_layers(model, AdaLoraLayer)
    for n, p in model.classifier.named_parameters():
        p.requires_grad = True
        print(n)
    return model, adapter_layers


def vit_b32_adalora(r=16, adapter_name="dlrt", n_classes=10):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    peft_config = peft.tuners.adalora.AdaLoraConfig(
        target_r=4,
        init_r=r,
        tinit=100,
        tfinal=50000,
        total_step=100000,
        deltaT=100,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.1,
        inference_mode=False,
        r=r,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=[
            "query",
            "key",
            "value",
            # "classifier",
            "intermediate.dense",  # might throw error
            "output.dense",
        ],
        bias="all",
    )  #### just some layers

    model = AdaLoraModel(model, peft_config, adapter_name)
    for p in model.classifier.parameters():
        p.requires_grad = True

    adapter_layers = collect_lr_layers(model, AdaLoraLayer)

    return model, adapter_layers


def vit_b32_lora(r=16, adapter_name="lora", n_classes=10):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        torch_dtype=torch.float32,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    peft_config = peft.tuners.lora.LoraConfig(
        inference_mode=False,
        r=r,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=[
            "query",
            "key",
            "value",
            # "classifier",
            "intermediate.dense",  # might throw error
            "output.dense",
        ],
        bias="none",
    )  #### just some layers

    model = peft.tuners.lora.LoraModel(model, peft_config, adapter_name)
    for p in model.classifier.parameters():
        p.requires_grad = True
    return model


def test(r=16, adapter_name="dlrt"):
    class test_model(torch.nn.Module):
        def __init__(self):
            super(test_model, self).__init__()
            self.flatten = torch.nn.Flatten()
            self.seq1 = torch.nn.Linear(28 * 28, 100, bias=True)
            self.seq2 = torch.nn.Linear(100, 10, bias=True)
            self.nl = torch.nn.Tanh()

        def forward(self, x):
            return self.seq2(self.nl(self.seq1(self.flatten(x))))

    model = test_model()
    model.eval()
    peft_config = AdaLoraConfig(
        orth_reg_weight=0.0,
        inference_mode=False,
        r=r,
        lora_alpha=1,
        lora_dropout=0.0,
        target_modules=["seq1"],
        bias="all",
    )
    model = AdaLoraModel(model, peft_config, adapter_name)
    adapter_layers = collect_lr_layers(model, AdaLoraLayer)
    for p in model.seq2.parameters():
        p.requires_grad = True
    with torch.no_grad():
        for l in adapter_layers:
            l.lora_E[adapter_name].copy_(torch.abs(l.lora_E[adapter_name]))
    return model, adapter_layers
