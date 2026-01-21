import torch
import torchvision
import types
import random
import matplotlib.pyplot as plt
import numpy as np
import umap
### PART 1 IMPLEMENTATION

def part_1() -> None:
    # freeze and replacement with appropriate number of classes
    resnet152 = torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
    for param in resnet152.parameters():
        param.requires_grad = False
    resnet152.fc = torch.nn.Linear(resnet152.fc.in_features, 10)
    for param in resnet152.fc.parameters():
        param.requires_grad = True
    
    transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])

    train_data = torchvision.datasets.CIFAR10("./datasets",download=True,transform=transformation)
    test_data = torchvision.datasets.CIFAR10("./datasets",train=False,transform=transformation)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,pin_memory=True)

    # Hyperparameters
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 5

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet152.to(device=device)
    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet152.parameters()), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # If the machine you are running on doesn't have a gpu or is an older model
    # disable these
    print("COMPILE START")
    resnet152 = torch.compile(resnet152, mode="max-autotune")
    torch.set_float32_matmul_precision('high')

    # TODO: make this a seperate function and add the appropriate args instead of lazily adding the func here
    def run_epoch(loader, is_train=True):
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        total_grad_norm = 0.0  
        num_batches = len(loader)
        
        context = torch.no_grad() if not is_train else torch.enable_grad()
        
        with context:
            for data, targets in loader:
                data = data.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                
                predictions = resnet152(data)
                loss = loss_fn(predictions, targets)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    for param in resnet152.fc.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.detach().norm(2).item()

                    optimizer.step()

                total_loss += loss.detach()
                total_correct += (predictions.argmax(dim=1) == targets).sum()

        avg_grad = total_grad_norm / num_batches if is_train else 0.0
        return total_loss.item() / num_batches, total_correct.item() / len(loader.dataset), avg_grad

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, grad = run_epoch(train_loader, is_train=True)
        output_str = f"Train | Epoch: {epoch} | Acc: {train_acc:.4f} | Loss: {train_loss:.4f} | grad: {grad}\n"
        print(output_str)
        with open("part_1_metrics.txt",'a') as f:
            f.write(output_str)
            
        test_loss, test_acc, grad = run_epoch(test_loader, is_train=False)
        output_str = f"Test  | Epoch: {epoch} | Acc: {test_acc:.4f} | Loss: {test_loss:.4f} | grad: {grad}\n"
        print(output_str)
        with open("part_1_metrics.txt",'a') as f:
            f.write(output_str)
    
    torch.save(resnet152.state_dict(), 'resnet152_cifar10_weights.pth')


### PART 2 IMPLEMENTATION


def forward_without_identity(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out += identity
        out = self.relu(out)

        return out

def apply_no_identity_modifications(model, seed=42):
    random.seed(seed) 
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    
    for layer in layers:
        for block in layer:
            if random.random() > 0.1:
                block.forward = types.MethodType(forward_without_identity, block)
    return model

def part_2():
    resnet152 = torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
    resnet152.eval()
    orig_model = torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
    orig_model.eval()

    layers = [resnet152.layer1, resnet152.layer2, resnet152.layer3, resnet152.layer4]

    for layer in layers:
        for block in layer:
        # it is kinda possible that none of these get replaced now that I think about it    
            if random.random() > 0.1:
                block.forward = types.MethodType(forward_without_identity, block)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = resnet152(dummy_input)
        output2 = orig_model(dummy_input)
    res =  torch.allclose(output,output2)
    
    if res is True:
        print("Congrats! you got the extremely unlikely case where none of the modules were replaced!")
        print("Please purchase a lottery ticket")
        return

    print("Tensor Match with original model:",res)
    resnet152.train()

    # Rest of this is literally just coppied from part 1
    for param in resnet152.parameters():
        param.requires_grad = False
    resnet152.fc = torch.nn.Linear(resnet152.fc.in_features, 10)
    for param in resnet152.fc.parameters():
        param.requires_grad = True
    
    transformation = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])

    train_data = torchvision.datasets.CIFAR10("./datasets",download=True,transform=transformation)
    test_data = torchvision.datasets.CIFAR10("./datasets",train=False,transform=transformation)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False,pin_memory=True)

    # Hyperparameters
    learning_rate = 0.01
    batch_size = 32
    num_epochs = 5

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resnet152.to(device=device)
    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet152.parameters()), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # If the machine you are running on doesn't have a gpu or is an older model
    # disable these
    print("COMPILE START")
    resnet152 = torch.compile(resnet152, mode="max-autotune")
    torch.set_float32_matmul_precision('high')

    # TODO: make this a seperate function and add the appropriate args instead of lazily adding the func here
    def run_epoch(loader, is_train=True):
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        total_grad_norm = 0.0 # Track gradient magnitude
        num_batches = len(loader)
        
        context = torch.no_grad() if not is_train else torch.enable_grad()
        
        with context:
            for data, targets in loader:
                data = data.to(device=device, non_blocking=True)
                targets = targets.to(device=device, non_blocking=True)
                
                predictions = resnet152(data)
                loss = loss_fn(predictions, targets)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    for param in resnet152.fc.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.detach().norm(2).item()

                    optimizer.step()

                total_loss += loss.detach()
                total_correct += (predictions.argmax(dim=1) == targets).sum()

        avg_grad = total_grad_norm / num_batches if is_train else 0.0
        return total_loss.item() / num_batches, total_correct.item() / len(loader.dataset), avg_grad
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, grad = run_epoch(train_loader, is_train=True)
        output_str = f"Train | Epoch: {epoch} | Acc: {train_acc:.4f} | Loss: {train_loss:.4f} | grad: {grad}\n"
        print(output_str)
        with open("part_2_metrics.txt",'a') as f:
            f.write(output_str)
            
        test_loss, test_acc, grad = run_epoch(test_loader, is_train=False)
        output_str = f"Test  | Epoch: {epoch} | Acc: {test_acc:.4f} | Loss: {test_loss:.4f} | grad: {grad}\n"
        print(output_str)
        with open("part_2_metrics.txt",'a') as f:
            f.write(output_str)

    torch.save(resnet152.state_dict(), 'resnet152_no_identity_weights.pth')

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") 
        new_state_dict[name] = v
    return new_state_dict

def part_3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = torchvision.models.resnet152(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
    resnet_state_dict = torch.load('resnet152_cifar10_weights.pth', map_location=device)
    # since the model was saved as a compiled model the state dict needs a bit of fixing to be used
    resnet_state_dict = clean_state_dict(resnet_state_dict)
    resnet.load_state_dict(resnet_state_dict)
    resnet.to(device).eval()

    resnet_id_removal = torchvision.models.resnet152()
    resnet_id_removal.fc = torch.nn.Linear(resnet_id_removal.fc.in_features, 10)
    resnet_no_identity = apply_no_identity_modifications(resnet_id_removal, seed=42)
    no_identity_state_dict = torch.load('resnet152_no_identity_weights.pth', map_location=device)
    no_identity_state_dict = clean_state_dict(no_identity_state_dict)
    resnet_no_identity.load_state_dict(no_identity_state_dict)
    resnet_no_identity.to(device).eval()

    
    models = {
        "ResNet": resnet,
        "No-Identity ResNet": resnet_no_identity
    }

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ])
    test_data = torchvision.datasets.CIFAR10("./datasets", train=False, transform=transformation)
    indices = list(range(250)) 
    subset_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_data, indices), batch_size=100, shuffle=False
    )

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            #(batch, channels, h, w) -> (batch, features)
            activations[name] = torch.flatten(output, 1).detach().cpu().numpy()
        return hook

    layer_names = {
        "Beginning": "layer1",
        "Middle": "layer3",
        "Final": "avgpool"
    }

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    plt.subplots_adjust(hspace=0.3)

    for row, (model_name, model) in enumerate(models.items()):
        model.eval()
        
        hooks = []
        hooks.append(model.layer1.register_forward_hook(get_activation("Beginning")))
        hooks.append(model.layer3.register_forward_hook(get_activation("Middle")))
        hooks.append(model.avgpool.register_forward_hook(get_activation("Final")))

        all_labels = []
        captured_features = {"Beginning": [], "Middle": [], "Final": []}

        with torch.no_grad():
            for images, labels in subset_loader:
                images = images.to(device)
                _ = model(images)
                all_labels.extend(labels.numpy())
                for k in layer_names.keys():
                    captured_features[k].append(activations[k])

        for h in hooks:
            h.remove()

        for col, layer_key in enumerate(layer_names.keys()):
            feat = np.concatenate(captured_features[layer_key], axis=0)
            
            print(f"{model_name}:{layer_key}")
            reducer = umap.UMAP(n_neighbors=30, n_components=2)
            embedded = reducer.fit_transform(feat)

            ax = axes[row, col]
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=all_labels, cmap='tab10', s=10, alpha=0.7)
            ax.set_title(f"{model_name}: {layer_key}")
            
            if row == 0 and col == 2:
                ax.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("feature_visualization.png")

def part_4() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data = torchvision.datasets.OxfordIIITPet("./datasets", split="trainval", download=True, transform=transformation)
    test_data = torchvision.datasets.OxfordIIITPet("./datasets", split="test", download=True, transform=transformation)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

    def get_model(pretrained=True, fine_tune_all=True):
        weights = "ResNet152_Weights.IMAGENET1K_V1" if pretrained else None
        model = torchvision.models.resnet152(weights=weights)
        # 37 Classes in total
        model.fc = torch.nn.Linear(model.fc.in_features, 37)
        
        if not fine_tune_all:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True
        
        return model.to(device)

    def train_experiment(model, name, epochs=5):
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_correct = 0.0, 0.0
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_correct += (output.argmax(1) == targets).sum().item()
            
            train_acc = total_correct / len(train_loader.dataset)
            
            model.eval()
            test_loss, test_correct = 0.0, 0.0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    output = model(data)
                    test_loss += loss_fn(output, targets).item()
                    test_correct += (output.argmax(1) == targets).sum().item()
            
            test_acc = test_correct / len(test_loader.dataset)
            log = f"Run: {name} | Epoch: {epoch} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}\n"
            print(log)
            with open("part_4_metrics.txt", "a") as f:
                f.write(log)

    experiments = [
        {"name": "Pretrained Full Training", "pretrained": True, "fine_tune_all": True},
        {"name": "Random init", "pretrained": False, "fine_tune_all": True},
        {"name": "Final Layer only", "pretrained": True, "fine_tune_all": False}
    ]

    for exp in experiments:
        model = get_model(pretrained=exp["pretrained"], fine_tune_all=exp["fine_tune_all"])
        train_experiment(model, exp["name"])


if __name__ == "__main__":

    # part_1()
    # part_2()
    # part_3()
    part_4()