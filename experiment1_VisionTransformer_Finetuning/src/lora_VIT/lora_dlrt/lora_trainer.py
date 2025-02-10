import torch

from collections import Counter


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
        # self.scaler = torch.cuda.amp.GradScaler()
        self.best_accuracy = 0.0

    def compute_lr_budget(self, rank_budget):
        # Step 0: Get the singular values of the layers
        singular_values_list = [l.get_singular_values() for l in self.dlrt_layers]
        # Step 1: Flatten the list and associate each singular value with its corresponding layer index
        flattened_singular_values = [
            (s_val, layer_idx)
            for layer_idx, s_vals in enumerate(singular_values_list)
            for s_val in s_vals
        ]
        # Step 2: Sort the flattened list by singular values in descending order
        flattened_singular_values.sort(key=lambda x: x[0], reverse=True)
        # Step 3: Extract the top K singular values
        top_singular_values = flattened_singular_values[:rank_budget]
        # Step 4: Count the occurrences of each layer index in the top K singular values
        layer_indices = [layer_idx for _, layer_idx in top_singular_values]
        # Step 5: Count the occurrences of each layer index
        layer_counts = Counter(
            {layer_idx: 0 for layer_idx in range(len(singular_values_list))}
        )
        layer_counts.update(layer_indices)
        # Step 6: Truncate the layers based on the layer counts
        # print(layer_counts)
        return layer_counts

    def train_classic(
        self,
        num_epochs,
        criterion,
        optimizer,
        scheduler,
        dlrt_lr=5e-2,
        tau=0.1,
        args=None,
        coeff_steps=1,
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

            watermark = "dlrt_model{}_lr{}_dlrtTau{}_epochs{}".format(
                args.net_name,
                args.lr,
                args.tau,
                num_epochs,
            )
            wandb.init(
                project="Adalora3_benchmark{}".format(args.dataset_name, args.net_name),
                name=watermark,
            )
            wandb.config.update(args)
            wandb.watch(self.model)

        # Training loop
        for epoch in range(num_epochs):

            loss = self.run_classic_integrator(
                optimizer=optimizer,
                coeff_steps=coeff_steps,
                dlrt_lr=dlrt_lr,
                tau=tau,
                criterion=criterion,
                args=args,
            )
            scheduler.step()
            dlrt_lr = scheduler.get_last_lr()[0]
            # evaluate model on test date

            print([l.r[l.adapter_name] for l in self.dlrt_layers])
            print("sum ranks, epoch ", epoch)
            print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))
            print("loss, epoch ", loss)

            val_acc, params = self.test_model()
            print("val_acc, epoch ", val_acc)

            if args.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "val_accuracy": val_acc,
                        "best val acc": self.best_accuracy,
                        "learning_rate": dlrt_lr,
                        "rank ": [l.r[l.adapter_name] for l in self.dlrt_layers],
                        "params ": params,
                    }
                )
        print("start finetuning")
        for epoch in range(1):
            loss = self.run_classic_coeff_ft(
                optimizer=optimizer,
                criterion=criterion,
            )
            scheduler.step()
            dlrt_lr = scheduler.get_last_lr()[0]
            # evaluate model on test date

            print([l.r[l.adapter_name] for l in self.dlrt_layers])
            print("sum ranks")
            print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))

            val_acc, params = self.test_model()

            if args.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "val_accuracy": val_acc,
                        "best val acc": self.best_accuracy,
                        "learning_rate": dlrt_lr,
                        "rank ": [l.r[l.adapter_name] for l in self.dlrt_layers],
                        "params ": params,
                    }
                )
        return 0

    def run_classic_integrator(
        self,
        optimizer,
        coeff_steps,
        dlrt_lr,
        tau,
        criterion,
        args=None,
    ):
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
            optimizer.zero_grad()
            loss.backward()

            ################ update entire network without low-rank coefficients ################
            #### assuming to pass to the optimizer just the parameters for which one wants really to do the step #####
            # self.scaler.step(optimizer = optimizer)
            if coeff_steps > 0:
                if batch_idx % (coeff_steps + 1) == 0:
                    for l in self.dlrt_layers:  # augmentation
                        l.step(dlrt_lr, tau, "dlrt")
                else:  # standard S step
                    optimizer.step()

                if batch_idx % (coeff_steps + 1) == int(3 * coeff_steps / 4):
                    for l in self.dlrt_layers:  # Truncation
                        l.Truncate(tau, "dlrt")

                    # if (batch_idx + 1) % 100 == 0:  # ranks after truncation
                    # print("after truncation")
                    # print([l.r[l.adapter_name] for l in self.dlrt_layers])

            else:  # standard algorithm
                for l in self.dlrt_layers:  # augmentation
                    l.step(dlrt_lr, tau, "dlrt")
                for l in self.dlrt_layers:  # truncation
                    l.Truncate(tau, "dlrt")
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
                print([l.r[l.adapter_name] for l in self.dlrt_layers])
                print("sum ranks")
                print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))
        return loss

    def run_classic_coeff_ft(
        self,
        optimizer,
        criterion,
    ):
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
            optimizer.zero_grad()
            loss.backward()

            ################ update entire network without low-rank coefficients ################
            #### assuming to pass to the optimizer just the parameters for which one wants really to do the step #####
            # self.scaler.step(optimizer = optimizer)

            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"FT Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
                print([l.r[l.adapter_name] for l in self.dlrt_layers])
                print("sum ranks")
                print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))

        return loss

    def train_budget(
        self,
        num_epochs,
        criterion,
        optimizer,
        scheduler,
        dlrt_lr=5e-2,
        tau=0.1,
        args=None,
        coeff_steps=1,
        rank_budget=500,
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

            watermark = "dlrt_budget_model{}_lr{}_dlrtTau{}_epochs{}".format(
                args.net_name,
                args.lr,
                args.tau,
                num_epochs,
            )
            wandb.init(
                project="Adalora2_benchmark{}".format(args.dataset_name, args.net_name),
                name=watermark,
            )
            wandb.config.update(args)
            wandb.watch(self.model)

        # Training loop
        for epoch in range(num_epochs):

            loss = self.run_budget_integrator(
                optimizer=optimizer,
                coeff_steps=coeff_steps,
                dlrt_lr=dlrt_lr,
                tau=tau,
                criterion=criterion,
                args=args,
                rank_budget=rank_budget,
            )

            scheduler.step()
            dlrt_lr = scheduler.get_last_lr()[0]
            # evaluate model on test date
            print([l.r[l.adapter_name] for l in self.dlrt_layers])
            print("sum ranks")
            print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))
            val_acc, params = self.test_model()

            if args.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "val_accuracy": val_acc,
                        "best val acc": self.best_accuracy,
                        "learning_rate": dlrt_lr,
                        "rank ": [l.r[l.adapter_name] for l in self.dlrt_layers],
                        "params ": params,
                    }
                )

        for epoch in range(1):
            loss = self.run_classic_coeff_ft(
                optimizer=optimizer,
                criterion=criterion,
            )
            scheduler.step()
            dlrt_lr = scheduler.get_last_lr()[0]
            # evaluate model on test date

            print([l.r[l.adapter_name] for l in self.dlrt_layers])
            print("sum ranks")
            print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))

            val_acc, params = self.test_model()

            if args.wandb:
                wandb.log(
                    {
                        "loss train": float(loss.item()),
                        "val_accuracy": val_acc,
                        "best val acc": self.best_accuracy,
                        "learning_rate": dlrt_lr,
                        "rank ": [l.r[l.adapter_name] for l in self.dlrt_layers],
                        "params ": params,
                    }
                )

    def run_budget_integrator(
        self,
        optimizer,
        coeff_steps,
        dlrt_lr,
        tau,
        criterion,
        rank_budget,
        args=None,
    ):
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
            optimizer.zero_grad()
            loss.backward()

            ################ update entire network without low-rank coefficients ################
            #### assuming to pass to the optimizer just the parameters for which one wants really to do the step #####
            # self.scaler.step(optimizer = optimizer)
            if coeff_steps > 0:
                if batch_idx % (coeff_steps + 1) == 0:
                    for l in self.dlrt_layers:  # augmentation
                        l.step(dlrt_lr, tau, "dlrt")
                else:  # standard S step
                    optimizer.step()

                if batch_idx % (coeff_steps + 1) == int(3 * coeff_steps / 4):
                    # print("++++++")
                    # print([l.r[l.adapter_name] for l in self.dlrt_layers])

                    layer_counts = self.compute_lr_budget(
                        rank_budget
                    )  # compute low rank budget
                    # print(layer_counts)
                    tmp_l_count = 0
                    for l in self.dlrt_layers:
                        # print(l.r["dlrt"])
                        # print(layer_counts[tmp_l_count])
                        # print("---")
                        l.budget_truncate(layer_counts[tmp_l_count], "dlrt")
                        tmp_l_count += 1

                    # if (batch_idx + 1) % 100 == 0:  # ranks after truncation
                    # print("after truncation")
                    # print([l.r[l.adapter_name] for l in self.dlrt_layers])

            else:  # standard algorithm
                for l in self.dlrt_layers:  # augmentation
                    l.step(dlrt_lr, tau, "dlrt")

                layer_counts = self.compute_lr_budget(
                    rank_budget
                )  # compute low rank budget
                tmp_l_count = 0
                for l in self.dlrt_layers:
                    # print(l.r["dlrt"])
                    # print(layer_counts[tmp_l_count])
                    # print("---")
                    l.budget_truncate(layer_counts[tmp_l_count], "dlrt")

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}"
                )
                print([l.r[l.adapter_name] for l in self.dlrt_layers])
                print("sum ranks")
                print(sum([l.r[l.adapter_name] for l in self.dlrt_layers]))
        return loss

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
                # with torch.cuda.amp.autocast():
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
            params = 0
            for n, m, r in [
                (l.in_features, l.out_features, l.r[l.adapter_name])
                for l in self.dlrt_layers
            ]:
                params += m * r + n * r + r ^ 2
        return accuracy, params
