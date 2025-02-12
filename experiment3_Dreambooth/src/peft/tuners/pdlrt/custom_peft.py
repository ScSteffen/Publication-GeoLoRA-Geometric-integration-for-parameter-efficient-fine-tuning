from .layer import LoRAParallelDLRT
from torch import nn
from .LoRAFullFT import LoRAFullFT


def get_custom_peft(
    model, peft_module, target_layer_names, rank, alpha, max_rank, tau, lora_dropout
):
    print(f"Custom peft module: {peft_module}")
    linear_replacements = []
    # trainable_params = [
    #    (name) for name, param in model.named_parameters() if param.requires_grad
    # ]
    # print(trainable_params)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(layer_name in name for layer_name in target_layer_names):
                linear_replacements.append((name, module))

            # Freeze all trainable weights
            # if "classifier" in name:
            #    for _, param in module.named_parameters():
            #        param.requires_grad = True
            # else:
        for _, param in module.named_parameters():
            param.requires_grad = False
    # trainable_params = [
    #  (name) for name, param in model.named_parameters() if param.requires_grad
    #
    # print(trainable_params)
    # print(linear_replacements)
    # exit(1)
    # Apply LORA
    # Replace layers after iteration
    lora_layers_DLRT = []
    for name, original_linear in linear_replacements:

        if name == "classifier":
            parent_module = model
            child_name = "classifier"
            lora_module = LoRAFullFT(
                original_linear, rank=original_linear.out_features, alpha=alpha
            )
            # print(original_linear.in_features, original_linear.out_features)
            # print(original_linear)
        else:
            if peft_module == "LoRAParallelDLRT":
                lora_module = LoRAParallelDLRT(
                    original_linear,
                    rank=rank,
                    alpha=alpha,
                    max_rank=max_rank,
                    tau=tau,
                    lora_dropout=lora_dropout,
                )
            else:
                raise ValueError(
                    "Invalid peft_module. Please choose from 'LoRABUGDLRT' or 'LoRAParallelDLRT'."
                )

            parent_name, child_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)

        setattr(parent_module, child_name, lora_module)
        lora_module = model.get_submodule(name)
        lora_layers_DLRT.append(lora_module)  # list all the lora layers

        # print([name for name, param in lora_module.named_parameters()])
        for name, param in lora_module.named_parameters():
            if (
                "lora_U" in name
                or "lora_S" in name
                or "lora_V" in name
                or "bias" in name
                or "lora_W" in name
            ):
                param.requires_grad = True  # LoRA layers to be trainable

        # if name == "classifier":
        #    for name, param in lora_module.named_parameters():
        #        if "lora_A" in name or "lora_B" in name or "bias" in name:
        #            param.requires_grad = True  # LoRA layers to be trainable

    # trainable_params = [
    #    (name) for name, param in model.named_parameters() if param.requires_grad
    # ]
    # print(trainable_params)
    # print(lora_layers_DLRT)
    # exit(1)
    return model, lora_layers_DLRT
