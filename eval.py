import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T
import open_clip
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import csv, json, os

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

def evaluate(model, dataloader, save_dir="results"):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    all_results = []
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.cuda()
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())
            for i in range(len(labels)):
                all_results.append({
                    "true_label": int(labels[i].item()),
                    "pred_label": int(preds[i].cpu().item()),
                    "prob_fake": float(probs[i, 1].cpu().item())
                })
    label_counts = Counter(y_true)
    metrics = {}
    if len(label_counts) < 2:
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        metrics = {"F1": f1, "Accuracy": acc}
    else:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        eer = fpr[np.nanargmin(np.abs(fpr + tpr - 1))]
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        metrics = {"AUC": auc, "EER": eer, "F1": f1, "Accuracy": acc}
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()
    # 儲存 metrics
    with open(os.path.join(save_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # 儲存預測細節
    with open(os.path.join(save_dir, "predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true_label", "pred_label", "prob_fake"])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    with open(os.path.join(save_dir, "predictions.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {save_dir}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_root = "data/Real_youtube"
    nt_root = "data/NeuralTextures"
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]),
    ])
    real_paths = collect_image_paths(real_root)
    split_idx = int(0.9 * len(real_paths))
    real_test = real_paths[split_idx:]
    test_real = DeepfakeDataset(real_test, label=0, transform=transform)
    test_fake = DeepfakeDataset(collect_image_paths(nt_root), label=1, transform=transform)
    test_loader = DataLoader(ConcatDataset([test_real, test_fake]), batch_size=32, shuffle=False)
    class_names = ['real', 'fake']
    model = PromptTunedCLIP(class_names=class_names, device=device).to(device)
    # 載入訓練好的模型
    model.load_state_dict(torch.load("checkpoints/promptclip.pt", map_location=device))
    evaluate(model, test_loader, save_dir="results")

if __name__ == "__main__":
    main()