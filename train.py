import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import open_clip
import random
from tqdm import tqdm
from collections import Counter
import os

# 假設 collect_image_paths, DeepfakeDataset 已定義

class PromptTunedCLIP(nn.Module):
    def __init__(self, class_names, prompt_len=5, device="cuda"):
        super().__init__()
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.prompt_len = prompt_len
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.embedding_dim = self.model.token_embedding.embedding_dim
        self.prompt_embed = nn.Parameter(torch.randn(self.n_classes, prompt_len, self.embedding_dim))
        for param in self.model.parameters():
            param.requires_grad = False
        with torch.no_grad():
            self.tokenized_prompts = self.tokenizer(
                [f"a photo of a {name} face" for name in class_names]
            ).to(self.device)
        self.linear = nn.Linear(self.model.visual.output_dim, self.n_classes)

    def forward(self, images):
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.linear(image_features)
        return logits

def train(model, dataloader, epochs=5, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam([
        {'params': model.prompt_embed, 'lr': lr},
        {'params': model.linear.parameters(), 'lr': lr}
    ])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images, labels = images.cuda(), labels.cuda()
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

def main():
    torch.manual_seed(42)
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_root = "data/Real_youtube"
    fake_root = "data/FaceSwap"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
    ])
    real_paths = collect_image_paths(real_root)
    random.shuffle(real_paths)
    split_idx = int(0.9 * len(real_paths))
    real_train = real_paths[:split_idx]
    train_real = DeepfakeDataset(real_train, label=0, transform=transform)
    train_fake = DeepfakeDataset(collect_image_paths(fake_root), label=1, transform=transform)
    train_loader = DataLoader(ConcatDataset([train_real, train_fake]), batch_size=32, shuffle=True)
    class_names = ['real', 'fake']
    model = PromptTunedCLIP(class_names=class_names, device=device).to(device)
    train(model, train_loader, epochs=10, lr=1e-3)
    # 儲存模型
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/promptclip.pt")
    print("Model saved to checkpoints/promptclip.pt")

if __name__ == "__main__":
    main()