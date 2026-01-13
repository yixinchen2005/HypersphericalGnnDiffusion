import torch
from models.ddim_bitdit import BitDit
from trainer_ner import Trainer
from data.ner.ner_dataset import NERDataset1D, Collator1D, LabelSet1D
from torch.utils.data import DataLoader
from tqdm import tqdm
import random, json, os
from prettytable import PrettyTable

class LEDTrainer(Trainer):
    def __init__(self, args):
        # Initialize base trainer (sets up device, logging, optimizer)
        super().__init__(args)
        
        # OVERRIDE: Load the Pre-trained Weights immediately
        if self.args.resume_path:
            print(f"Loading pre-trained diffusion weights from {self.args.resume_path}...")
            # We use strict=False in case you added new classifier-free guidance layers later
            self.model.load_state_dict(torch.load(self.args.resume_path), strict=True)
        else:
            raise ValueError("LED Training requires a pre-trained model path (args.resume_path)")

    # OVERRIDE: Get dataloader for the LABELED (3-column) dataset
    def _get_dataloader(self, mode: str, bsz: int):
        # We use the "labeled" data_type we added to the Dataset class
        dataset = NERDataset1D(self.args.dataset, mode, self.label_set, data_type="labeled")
        
        # We need a collator that handles the 3rd column (targets_old vs targets_new)
        # You might need to slightly update Collator1D or just ensure NERDataset1D 
        # returns (sentence, old, new) and Collator pads all of them.
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            num_workers=self.args.num_workers,
            shuffle=(mode == "train"),
            collate_fn=self.collate_fn 
        )
        return dataloader
    
    def _configure_dataloaders(self):
        print("Configuring LED Dataloaders (Split Strategy)...")
        
        # 1. Load the single 'labeled.json' file manually
        labeled_path = os.path.join(os.getcwd(), 'data', 'ner', self.args.dataset, 'labeled', 'labeled.json')
        if not os.path.exists(labeled_path):
            raise FileNotFoundError(f"Please generate {labeled_path} using convert.py first.")
            
        with open(labeled_path, 'r') as f:
            all_data = json.load(f)
            
        # 2. Split 400 items -> Train (320) / Dev (80)
        # Check if we should shuffle (set seed for reproducibility)
        random.seed(42) 
        random.shuffle(all_data)
        
        split_idx = int(0.8 * len(all_data))
        train_data = all_data[:split_idx]
        dev_data = all_data[split_idx:]
        
        print(f"Split {len(all_data)} items into {len(train_data)} Train and {len(dev_data)} Dev.")

        # 3. Create Datasets using the IN-MEMORY data_list
        # Note: We pass data_type='labeled' so __getitem__ knows to return 3 columns
        train_dataset = NERDataset1D(self.args.dataset, 'train', self.label_set, data_type='labeled', data_list=train_data)
        dev_dataset = NERDataset1D(self.args.dataset, 'dev', self.label_set, data_type='labeled', data_list=dev_data)
        
        # 4. Create Loaders
        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )
        
        self.dev_dataloader = DataLoader(
            dev_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers
        )
        
        # 5. Handle Test (Reuse Dev or keep None)
        # We set it to dev_dataloader so eval_epoch('test') doesn't crash
        self.test_dataloader = self.dev_dataloader
        
        self.steps = len(self.train_dataloader) * self.args.max_epochs

    # OVERRIDE: The "Hijacking" Training Step
    def _step(self, batch):
        # Unpack the 4 items from the LED Dataloader
        input_ids, attention_mask, targets_old, targets_new = [x.to(self.device) for x in batch]
        
        loss = self.model(
            input_ids, 
            attention_mask, 
            seq_labels=targets_new,   # The Ground Truth
            cond_labels=targets_old     # The Hint
        )
        return loss
    
    def eval_step(self, pred_labels, targets_old, targets_new):
        """
        Custom LED Evaluation Step.
        Returns counts for:
        - True Errors (in dataset)
        - Detected Errors (model changed the label)
        - Correctly Detected (intersection)
        - Correctly Fixed (model changed to the specific correct label)
        """
        bsz = pred_labels.shape[0]
        # Create mask for valid tokens (ignore padding)
        labels_mask = targets_new != 0 # "[PAD]"
        
        # Counters
        n_true_errors = 0      # How many errors actually exist (Old != New)
        n_detected_errors = 0  # How many the model claimed exist (Old != Pred)
        n_correct_detect = 0   # Intersection (TP for Detection)
        n_correct_fix = 0      # Intersection + Correct Value (TP for Correction)
        
        n_stable_tokens = 0    # Tokens that were correct in Old
        n_false_alarms = 0     # Tokens that were correct but model changed them

        for i in range(bsz):
            # Extract valid sequences based on mask
            mask = labels_mask[i]
            t_old = targets_old[i][mask].cpu().numpy()
            t_new = targets_new[i][mask].cpu().numpy()
            t_pred = pred_labels[i][mask].cpu().numpy()
            
            # 1. Identify where the dataset has errors
            error_indices = (t_old != t_new)
            n_true_errors += error_indices.sum()
            
            # 2. Identify where model made changes (Detections)
            change_indices = (t_old != t_pred)
            n_detected_errors += change_indices.sum()
            
            # 3. Detection Success (Did we change the right tokens?)
            # Both must be True at the same index
            correct_detect_mask = error_indices & change_indices
            n_correct_detect += correct_detect_mask.sum()
            
            # 4. Correction Success (Did we change it to the RIGHT label?)
            # Must be a correct detection AND value must match new label
            # Note: t_pred == t_new implies t_pred != t_old for error indices
            correct_fix_mask = correct_detect_mask & (t_pred == t_new)
            n_correct_fix += correct_fix_mask.sum()
            
            # 5. Stability Check (False Alarms)
            # Where Old was Correct (Old == New)
            stable_indices = (t_old == t_new)
            n_stable_tokens += stable_indices.sum()
            
            # Did we change any of these?
            false_alarm_mask = stable_indices & (t_pred != t_old)
            n_false_alarms += false_alarm_mask.sum()

        return n_true_errors, n_detected_errors, n_correct_detect, n_correct_fix, n_stable_tokens, n_false_alarms

    @torch.no_grad()
    def eval_epoch(self, mode: str):
        self.model.eval() # <--- Triggers self.training = False in BitDit
        dataloader = self.dev_dataloader if mode == 'dev' else self.test_dataloader
        
        # Accumulate raw counts
        total_metrics = [0, 0, 0, 0, 0, 0]
        
        tqdm_loop = tqdm(dataloader, desc=f'LED {mode}')
        for batch in tqdm_loop:
            # We need targets for comparison
            _, _, targets_old, targets_new = [x.to(self.device) for x in batch]
            
            # Call _step: Since self.training is False, this returns (results, path_x)
            results, _ = self._step(batch)
            pred_labels = results
            
            # Calculate batch metrics
            batch_metrics = self.eval_step(pred_labels, targets_old, targets_new)
            
            # Sum up
            for i in range(6):
                total_metrics[i] += batch_metrics[i]

        # Use the pretty printer
        precision, recall, f1 = self._print_res(total_metrics)
        
        return precision, recall, f1
    
    def _print_res(self, metrics):
        """
        metrics: tuple (true_errors, detected, correct_detect, correct_fix, stable_toks, false_alarms)
        """
        n_true, n_det, n_tp_det, n_tp_fix, n_stable, n_false = metrics
        
        # Calculation
        det_prec = n_tp_det / n_det if n_det > 0 else 0.0
        det_rec = n_tp_det / n_true if n_true > 0 else 0.0
        det_f1 = 2 * (det_prec * det_rec) / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0
        
        cor_prec = n_tp_fix / n_det if n_det > 0 else 0.0
        cor_rec = n_tp_fix / n_true if n_true > 0 else 0.0
        cor_f1 = 2 * (cor_prec * cor_rec) / (cor_prec + cor_rec) if (cor_prec + cor_rec) > 0 else 0.0
        
        stability = 1.0 - (n_false / n_stable) if n_stable > 0 else 0.0

        # Pretty Table
        table = PrettyTable()
        table.title = "Label Error Detection Results"
        table.field_names = ["Metric", "Precision", "Recall", "F1 Score", "Count / Info"]
        
        table.add_row(["Detection", f"{det_prec:.4f}", f"{det_rec:.4f}", f"{det_f1:.4f}", f"Found {n_det}/{n_true} Errors"])
        table.add_row(["Correction", f"{cor_prec:.4f}", f"{cor_rec:.4f}", f"{cor_f1:.4f}", f"Fixed {n_tp_fix}/{n_true} Perfectly"])
        table.add_row(["Stability", "-", "-", "-", f"False Alarms: {n_false}"])
        
        print(table)
        print(f"Stability Score: {stability:.4f}")
        
        return cor_prec, cor_rec, cor_f1