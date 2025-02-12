import torch


def set_debug_apis(state: bool = False):
    torch.autograd.profiler.profile(enabled=state)
    torch.autograd.profiler.emit_nvtx(enabled=state)
    torch.autograd.set_detect_anomaly(mode=state)


class LoraTrainer:
    def __init__(self, model, lora_layers, train_loader, test_loader):
        """Constructs trainer which manages and trains neural network
        Args:
            net_architecture: Dictionary of the network architecture. Needs keys 'type' and 'dims'. Low-rank layers need key 'rank'.
            train_loader: loader for training data
            test_loader: loader for test data
        """

        # torch.manual_seed(0)

        # Initialize the model
        self.model = model

        # find all ids of dynamical low-rank layers, since these layer require two steps
        self.dlrt_layers = lora_layers

        # store train and test data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = list(self.dlrt_layers[0].parameters())[0].device
        set_debug_apis(False)
        self.scaler = torch.cuda.amp.GradScaler()
        self.best_accuracy = 0.0

    def train(
        self,
        num_epochs,
        criterion,
        optimizer,
        scheduler,
        dlrt_lr=5e-2,
        tau=0.1,
        args=None,
    ):
        """Trains neural network for specified number of epochs with specified learning rate
        Args:
            num_epochs: number of epochs for training
            learning_rate: learning rate for optimization method
            optimizer_type: used optimizer. Use Adam for vanilla training.
            criterion : Loss function to use
            optimizer : optimizer containing just the weights that has to be updated outside the DLRT routine
        """
        # Define the loss function and optimizer. Optimizer is only needed to set all gradients to zero.

        # torch.manual_seed(0)

        if args.wandb == 1:
            import wandb

            watermark = "model{}_lr{}_dlrtTau{}_epochs{}".format(
                args.net_name,
                args.lr,
                args.tau,
                num_epochs,
            )
            wandb.init(
                project="ParallelDLRTLoRa_benchmark{}_model_{}".format(
                    args.dataset_name, args.net_name
                ),
                name=watermark,
            )
            wandb.config.update(args)
            wandb.watch(self.model)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                for p in self.model.parameters():
                    if p.requires_grad:
                        p.grad = None
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast():
                    out = self.model(data)
                    out = out.logits if hasattr(out, "logits") else out
                    loss = criterion(out, targets)
                # self.scaler.scale(loss).backward() # TODO: Autoscaler makes problems in the DLRT STEP
                loss.backward()

                ################ update entire network without low-rank coefficients ################
                #### assuming to pass to the optimizer just the parameters for which one wants really to do the step #####
                # self.scaler.step(optimizer = optimizer)
                optimizer.step()

                for l in self.dlrt_layers:
                    l.step(dlrt_lr, tau, "dlrt")
                # self.scaler.update()
                # print progress
                if (batch_idx + 1) % 300 == 0:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                    )
            scheduler.step()
            # evaluate model on test date
            val_acc = self.test_model()

            if self.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "val_accuracy": val_acc,
                        "best val acc": self.best_accuracy,
                        "learning_rate": dlrt_lr,
                        "rank ": [l.r[l.adapter_name] for l in self.dlrt_layers],
                    }
                )

    def test_model(self):
        """Prints the model's accuracy on the test data"""
        # Test the model
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    outputs = outputs.logits if hasattr(outputs, "logits") else outputs
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            accuracy = 100 * correct / total
            print(f"Accuracy of the network on the test images: {accuracy}%")

            print(f"ranks: {[l.r[l.adapter_name] for l in self.dlrt_layers]}")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
        return accuracy
