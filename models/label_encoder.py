from typing import Dict, Optional, Tuple
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

class HeteroLabelEmbeddingGNN(nn.Module):
    def __init__(
        self,
        num_labels: int = 13,
        hidden_dim: int = 64,      # Reduced for smaller output space
        diffusion_dim: int = 4,    # Matched to DiffusionSL bit-equivalence
        dropout: float = 0.1,
        depth: int = 2
    ):
        super().__init__()
        self.num_labels = num_labels
        
        # Trainable latent parameters
        self.label_embeddings = nn.Parameter(
            torch.randn(num_labels, hidden_dim) * 0.5 
        )

        self.entry = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.edge_relations = {
            ('label', 'inside', 'label'): 0, 
            ('label', 'to_entity', 'label'): 1, 
            ('label', 'exit', 'label'): 2, 
            ('label', 'background', 'label'): 3
        }

        self.convs = nn.ModuleList([
            HeteroConv({
                rel: GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
                for rel in self.edge_relations.keys()
            }, aggr='mean') for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) # Re-introduced for stability
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.edge_relations) + 1),
        )

        # Simplified projection: No Tanh, just Linear to 4D
        self.to_diffusion = nn.Linear(hidden_dim, diffusion_dim)

    def forward(self, edge_index_dict=None):
        x = self.entry(self.label_embeddings)
        if edge_index_dict is None or len(edge_index_dict) == 0:
            return x
        
        x_dict = {'label': x}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # Apply dropout after normalization to prevent over-fitting on small graph
            x_dict = {'label': self.dropout(self.norm(F.gelu(x_dict['label'])))}
        return x_dict['label']
    
    def edge_logits(self, embeddings, src_idx, dst_idx):
        e_u = embeddings[src_idx]
        e_v = embeddings[dst_idx]
        # Added absolute difference to help decoder distinguish distance in space
        pair = torch.cat([e_u, e_v, torch.abs(e_u - e_v)], dim=-1)
        return self.edge_decoder(pair)
    
    def get_label_table(self, edge_index_dict):
        self.eval()
        with torch.no_grad():
            emb = self.forward(edge_index_dict)
            z = self.to_diffusion(emb)
            z = F.normalize(z, p=2, dim=-1) 
        return z
    
# --------------------- Training utilities ---------------------

def loss_edge_reconstruction(model, edge_index_dict, device, neg_ratio=1):
    model.to(device)
    embeddings = model(edge_index_dict).to(device)

    pos_src, pos_dst, pos_rel = [], [], []
    rel2id = {rel: i for i, rel in enumerate(model.edge_relations.keys())}
    
    # Relation weights to counter "O" (background) dominance
    # inside=2.0, to_entity=2.0, exit=1.0, background=0.1
    rel_weights = torch.tensor([2.0, 2.0, 1.0, 0.1, 1.0], device=device)

    for rel, edge_index in edge_index_dict.items():
        eid = rel2id[rel]
        if edge_index is None: continue
        src, dst = edge_index
        pos_src.append(src.to(device)); pos_dst.append(dst.to(device))
        pos_rel.append(torch.full((src.size(0),), eid, dtype=torch.long, device=device))

    if len(pos_src) == 0: return torch.tensor(0.0, device=device), {'acc': 0.0}

    pos_src, pos_dst, pos_rel = torch.cat(pos_src), torch.cat(pos_dst), torch.cat(pos_rel)
    
    # Negative sampling
    neg_src = pos_src.repeat(neg_ratio)
    neg_dst = torch.randint(0, model.num_labels, (pos_src.size(0) * neg_ratio,), device=device)
    neg_rel = torch.full((pos_src.size(0) * neg_ratio,), len(model.edge_relations), device=device, dtype=torch.long)

    src_idx = torch.cat([pos_src, neg_src])
    dst_idx = torch.cat([pos_dst, neg_dst])
    rel_idx = torch.cat([pos_rel, neg_rel])

    logits = model.edge_logits(embeddings, src_idx, dst_idx)
    # Apply relation weights to the CrossEntropy loss
    loss = F.cross_entropy(logits, rel_idx, weight=rel_weights)

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == rel_idx).float().mean().item()
    return loss, {'acc': acc}

# def loss_hyperspherical_uniformity(embeddings):
#     z = F.normalize(embeddings, p=2, dim=-1)
#     # RBF Kernel-based uniformity: rewards points for being far apart on the shell
#     dist_sq = torch.cdist(z, z, p=2).pow(2)
#     return torch.exp(-dist_sq).sum() / z.size(0)

def loss_hyperspherical_uniformity(embeddings):
    z = F.normalize(embeddings, p=2, dim=-1)
    # Use 1/dist logic (Riesz s-energy) to mimic electrostatic repulsion
    # This is more aggressive at maximizing the minimum distance between points
    dist = torch.cdist(z, z, p=2)
    # Add small epsilon and ignore diagonal (zeros)
    mask = ~torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    inv_dist = 1.0 / (dist[mask] + 1e-6)
    return inv_dist.mean()

def loss_spherical_angular_margin(embeddings, labels, intra_angle=0.2, inter_angle=0.8):
    def ent_type(l): return l[2:] if (l.startswith("B-") or l.startswith("I-")) else None

    # Ensure embeddings are on the sphere for the loss calculation
    z = F.normalize(embeddings, p=2, dim=-1)
    
    # Cosine similarity matrix [13, 13]
    cos_sim = torch.mm(z, z.t())
    
    loss_rep, loss_att = 0.0, 0.0
    c_rep, c_att = 1e-6, 1e-6

    for i in range(z.size(0)):
        ti = ent_type(labels[i])
        for j in range(i + 1, z.size(0)):
            tj = ent_type(labels[j])
            sim = cos_sim[i, j]

            if ti is not None and ti == tj:
                # Same entity: Maximize similarity (Similarity should be > 1 - intra_angle)
                loss_att += F.relu((1 - intra_angle) - sim)
                c_att += 1
            else:
                # Different entity: Minimize similarity (Similarity should be < 1 - inter_angle)
                loss_rep += F.relu(sim - (1 - inter_angle))
                c_rep += 1
                
    return (loss_att / c_att) + (loss_rep / c_rep)

def train(model, edge_index_dict, labels, device, epochs=1000):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    pbar = tqdm(range(1, epochs + 1), desc="Spherical GNN Training")
    for epoch in pbar:
        model.train()
        opt.zero_grad()
        
        # 1. Get hidden embeddings and project to Diffusion space (4D)
        emb_hidden = model(edge_index_dict)
        emb_diff = model.to_diffusion(emb_hidden) 
        
        # 2. Edge Reconstruction Loss (Structural Logic)
        # Keeps related labels (like B-PER/I-PER) in similar angular sectors
        recon_loss, metrics = loss_edge_reconstruction(model, edge_index_dict, device)
        
        # 3. Spherical Angular Margin Loss (Replaces aggressive Euclidean separation)
        # Forces intra-entity labels together and pushes different entities apart angularly
        # sep_loss = loss_spherical_angular_margin(
        #     emb_diff, labels, intra_angle=0.15, inter_angle=0.85
        # )
        sep_loss = loss_spherical_angular_margin(
            emb_diff, labels, intra_angle=0.6, inter_angle=1.2
        )
        
        # 4. Hyperspherical Uniformity Loss (Replaces Gaussian Normalization)
        # Ensures the 13 labels spread out across the shell rather than clustering
        uniform_loss = loss_hyperspherical_uniformity(emb_diff)
        
        # Weighted Total Loss: Prioritize the Spherical Geometry
        total_loss = (0.2 * recon_loss) + (3.0 * sep_loss) + (1.0 * uniform_loss)
        
        total_loss.backward()
        opt.step()
        
        pbar.set_postfix(
            recon=f"{recon_loss.item():.3f}",
            sep=f"{sep_loss.item():.3f}", 
            uni=f"{uniform_loss.item():.3f}", 
            acc=f"{metrics['acc']:.2f}"
        )

# --------------------- utils ---------------------

class TransitionParser:
    """Encapsulates NER transition logic to keep create_edge_index clean."""
    @staticmethod
    def get_etype(t, tgt_mask, prev_inside, prev_ent):
        is_subword = (t == "X" and tgt_mask)
        t_type = None if t in ["O", "X", "[SEP]", "[CLS]", "[PAD]"] else t[2:]

        if is_subword:
            return ("inside" if prev_inside else "background"), prev_inside, prev_ent
        if t == "X":
            return ("exit" if prev_inside else "background"), False, None
        if t.startswith("B-"):
            return "to_entity", True, t_type
        if prev_inside and t.startswith("I-") and t_type == prev_ent:
            return "inside", True, t_type
        if prev_inside and (t == "O" or t == "[SEP]"):
            return "exit", False, None
        return "background", False, None

def create_edge_index_dict(dataset, name, tokenizer, label_map, edge_types, max_len=128):
    path = f"data/ner/{dataset}/{name}.json"
    with open(path) as f:
        data = json.load(f)

    edge_buffers = {etype: [] for etype in edge_types}

    for item in tqdm(data, desc=f"Processing {name}"):
        enc = tokenizer(item["sentence"], is_split_into_words=True, truncation=True, max_length=max_len)
        word_ids = enc.word_ids(0)
        
        # Build sequence labels and masks efficiently
        seq_label, c_mask = [], []
        for i, wid in enumerate(word_ids):
            is_new_word = (wid is not None and (i == 0 or wid != word_ids[i-1]))
            seq_label.append(item["label"][wid] if is_new_word else "X")
            c_mask.append(0 if is_new_word or wid is None else 1)
        
        seq_label = ["[CLS]"] + seq_label[1:-1] + ["[SEP]"]
        lbl_ids = torch.tensor([label_map[l] for l in seq_label])
        
        # State machine for transitions
        prev_inside, prev_ent = False, None
        for i in range(1, len(seq_label)):
            etype, prev_inside, prev_ent = TransitionParser.get_etype(
                seq_label[i], c_mask[i], prev_inside, prev_ent
            )
            # Bi-directional edges
            edge = lbl_ids[i-1:i+1]
            edge_buffers[etype].append(torch.stack([edge, edge.flip(0)]))

    return { ( "label", k, "label" ): torch.cat(v, dim=1) for k, v in edge_buffers.items() if v }

def test_geometry(embeddings, labels, semantic_map, noise_levels=[0.1, 0.5, 1.0]):
    """
    Evaluates the codebook as a directional manifold on a hypersphere.
    """
    device = embeddings.device
    # Focus only on functional labels
    valid_ids = [i for i, l in enumerate(labels) if l not in ["[PAD]", "[CLS]", "[SEP]"]]
    
    # 1. FORCE Spherical Projection
    # This ensures we are testing the 'Hollow Shell' and not a Euclidean cloud
    # embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # 2. Metric: Hyperspherical Uniformity (Spread)
    # Measures if labels are well-distributed or bunched in one sector
    with torch.no_grad():
        all_sims = torch.mm(embeddings, embeddings.t())
        mask = ~torch.eye(len(labels), dtype=torch.bool, device=device)
        avg_cosine_sim = all_sims[mask].mean().item()
        # Ideal uniformity: Mean cosine similarity should be near or below 0
        uniformity_score = 1.0 - max(0, avg_cosine_sim) 

    ent_groups = {i: [labels.index(s) for s in semantic_map[labels[i]]] for i in valid_ids}
    
    results = {"global_uniformity": uniformity_score}
    print(f"\n--- Codebook Uniformity: {uniformity_score:.3f} (Ideal: > 0.8) ---")

    for sigma in noise_levels:
        # 3. Spherical Noise Simulation
        # We add noise and re-project to the shell to simulate Diffusion sampling
        noise = torch.randn(len(valid_ids), 1000, embeddings.size(1), device=device) * sigma
        z_noisy = F.normalize(embeddings[valid_ids].unsqueeze(1) + noise, p=2, dim=-1) # [V, 1000, D]
        
        # 4. Angular Nearest Neighbor Search (1 - Cosine Similarity)
        # Higher similarity = lower distance
        dists = 1 - torch.einsum('vsd,nd->vsn', z_noisy, embeddings)
        preds = dists.argmin(dim=-1) # [V, 1000]
        
        # 5. Metrics: Consistency and Locality
        correct_self = (preds == torch.tensor(valid_ids, device=device).unsqueeze(1)).float()
        correct_sem = correct_self.clone()
        for i, v_idx in enumerate(valid_ids):
            for sem_idx in ent_groups[v_idx]:
                correct_sem[i] += (preds[i] == sem_idx).float()

        # 6. Metric: Minimum Angular Gap (Angular Margin)
        # Calculates the radians between a label and its closest competitor
        angular_margins = []
        for i in valid_ids:
            # dot product of i with all others
            sims = torch.mm(embeddings[i:i+1], embeddings.t()).squeeze()
            sims[i] = -1.0 # Ignore self
            max_sim = sims.max().item()
            # Angle in radians: acos is safe since emb is normalized
            min_angle = torch.acos(torch.tensor(max_sim).clamp(-1, 1)).item()
            angular_margins.append(min_angle)

        results[sigma] = {
            "sem_locality": correct_sem.mean().item(),
            "self_consistency": correct_self.mean().item(),
            "min_angular_gap": sum(angular_margins) / len(angular_margins)
        }
        
        print(f"σ={sigma:3.1f} | Sem-Loc: {results[sigma]['sem_locality']:.3f} | "
              f"Consist: {results[sigma]['self_consistency']:.3f} | "
              f"Min-Gap: {results[sigma]['min_angular_gap']:.3f} rad")

    return results


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/yixin/workspace/huggingface/bert-large-cased")
    label_map = {
            "[PAD]": 0, "O": 1, "B-MISC": 2, "I-MISC": 3, "B-PER": 4, "I-PER": 5,
            "B-ORG": 6, "I-ORG": 7, "B-LOC": 8, "I-LOC": 9, "X": 10, "[CLS]": 11, "[SEP]": 12
        }
    edge_types = ['inside', 'to_entity', 'exit', 'background']
    semantic_similarities = {
            "B-PER": ["I-PER"], "I-PER": ["B-PER"],
            "B-ORG": ["I-ORG"], "I-ORG": ["B-ORG"],
            "B-LOC": ["I-LOC"], "I-LOC": ["B-LOC"],
            "B-MISC": ["I-MISC"], "I-MISC": ["B-MISC"],
            "O": [], "X": [], "[PAD]": [], "[CLS]": [], "[SEP]": []
        }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = device = 'cpu'
    edge_index_dict = create_edge_index_dict("twitter2015", "unlabeled", tokenizer, label_map, edge_types)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
    model = HeteroLabelEmbeddingGNN(hidden_dim=32, diffusion_dim=8)
    train(model=model, edge_index_dict=edge_index_dict, labels=list(label_map.keys()), device=device, epochs=500)
    z_table = model.get_label_table(edge_index_dict)
    torch.save(z_table, "label_codebook.pt")
    # z_table = torch.load("label_codebook.pt")
    test_geometry(z_table, list(label_map.keys()), semantic_similarities)