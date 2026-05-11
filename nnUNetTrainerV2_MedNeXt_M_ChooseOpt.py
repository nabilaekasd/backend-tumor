import sys
import os
import torch
import numpy as np

# Tambahkan path MedNeXt
sys.path.append('/home/nabila/backend-tumor/MedNeXt')

# Import kelas bawaan nnU-Net
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
# Import kelas MedNeXt asli
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_MedNeXt_M_kernel3

class nnUNetTrainerV2_MedNeXt_M_ChooseOpt(nnUNetTrainerV2_MedNeXt_M_kernel3, nnUNetTrainer):
    """
    Pick ONE optimizer for the entire run.
    """
    OPT_MAP = {
        "0": "adamw", "adamw": "adamw",
        "1": "rmsprop", "rmsprop": "rmsprop",
        "2": "adam", "adam": "adam"
    }

    def __init__(self, *args, **kwargs):
        # Panggil init dari MedNeXt
        nnUNetTrainerV2_MedNeXt_M_kernel3.__init__(self, *args, **kwargs)
        self.max_num_epochs = 200
        self.num_batches_per_epoch = 60
        self.num_val_batches_per_epoch = 30
        self.validate_every_n_epochs = 1
        self.fp16 = True
        self._opt_name = None
        self.initial_lr = 1e-3
        self.save_every_n_epochs = 20

    def _control_file_path(self):
        return os.path.join(self.output_folder, "OPTIMIZER.txt")

    def _read_choice(self):
        env_name = os.environ.get("OPTIMIZER", "") or os.environ.get("OPTIMIZER_ID", "")
        choice = env_name.strip()
        if not choice:
            p = self._control_file_path()
            if os.path.isfile(p):
                try:
                    with open(p, "r") as f:
                        choice = f.read().strip()
                except Exception:
                    choice = ""
        if not choice:
            choice = "adamw"
        choice_l = choice.lower()
        return self.OPT_MAP.get(choice_l, "adamw")

    def _make_optimizer(self, name: str):
        if name == "adamw":
            self.print_to_log_file("Using AdamW")
            return torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=1e-4)
        if name == "rmsprop":
            self.print_to_log_file("Using RMSprop")
            return torch.optim.RMSprop(self.network.parameters(), lr=self.initial_lr, momentum=0.9)
        if name == "adam":
            self.print_to_log_file("Using Adam")
            return torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=1e-5)
        raise ValueError(f"Unknown optimizer: {name}")

    def initialize(self, training=True, force_load_plans=False):
        # Panggil initialize dari MedNeXt
        nnUNetTrainerV2_MedNeXt_M_kernel3.initialize(self, training, force_load_plans)
        self._opt_name = self._read_choice()
        self.optimizer = self._make_optimizer(self._opt_name)

    def setup_DA_params(self):
        nnUNetTrainerV2_MedNeXt_M_kernel3.setup_DA_params(self)
        dap = self.data_aug_params

        # disable heavy augs
        dap['do_elastic'] = True
        dap['p_gamma'] = 0.2
        # light geo jitter
        deg = np.deg2rad(5)
        dap['rotation_x'] = (-deg, deg)
        dap['rotation_y'] = (-deg, deg)
        dap['rotation_z'] = (-deg, deg)
        dap['p_rot'] = 0.2
        dap['scale_range'] = (0.95, 1.05)
        dap['p_scale'] = 0.2

        # correct mirror keys
        dap['do_mirror'] = True
        dap['mirror_axes'] = (0, 1, 2)
        if 'mirror' in dap:
            del dap['mirror']

        # tone down pixel augs + keep loader light
        dap['p_blur'] = 0.1
        dap['p_gaussian_noise'] = 0.1
        dap['p_median_filtering'] = 0.0
        dap['p_brightness'] = 0.1
        dap['p_contrast'] = 0.1
        dap['num_threads'] = 2
        dap['num_cached_per_thread'] = 1
        for k in ('p_low_res_sample', 'p_motion_blur'):
            if k in dap:
                dap[k] = 0.0

    # helpers
    def _safe_to_tensor(self, arr, dev, *, is_target=False):
        if torch.is_tensor(arr):
            return arr.to(dev)

        if isinstance(arr, (list, tuple)):
            if len(arr) == 0:
                raise ValueError("Empty list/tuple received for batch.")
            if all(torch.is_tensor(x) for x in arr):
                try:
                    arr = torch.stack(arr, dim=0)
                except Exception:
                    arr = arr[0]
            else:
                try:
                    arr = np.stack(arr, axis=0)
                except Exception:
                    arr = arr[0]

        if isinstance(arr, np.ndarray):
            if is_target:
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64, copy=False)
            else:
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
            arr = np.ascontiguousarray(arr)
            t = torch.from_numpy(arr).to(dev)
        elif torch.is_tensor(arr):
            t = arr.to(dev)
        else:
            t = torch.as_tensor(arr, device=dev)

        if is_target and t.dtype != torch.int64:
            t = t.long()
        if not is_target and t.dtype != torch.float32:
            t = t.float()
        return t

    def _quick_fg_dice_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            elif isinstance(logits, dict) and 'out' in logits:
                logits = logits['out']

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            if target.ndim == preds.ndim + 1:
                target = target[:, 0]
            eps = 1e-8
            dices = []
            num_classes = int(getattr(self, "num_classes", probs.shape[1]))
            for c in range(1, num_classes):
                p = (preds == c)
                g = (target == c)
                inter = (p & g).sum().float()
                denom = p.sum().float() + g.sum().float()
                if denom > 0:
                    dices.append((2.0 * inter / (denom + eps)).item())
            return float(np.mean(dices)) if len(dices) > 0 else float("nan")

    def _estimate_train_dice_quick(self, max_batches: int = 2) -> float:
        if not hasattr(self, "tr_gen") or self.tr_gen is None:
            return float("nan")

        dev = next(self.network.parameters()).device
        self.network.eval()
        dices = []
        tried = 0
        try:
            with torch.no_grad():
                while tried < max_batches:
                    batch = next(self.tr_gen)
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 0:
                            break
                        batch = batch[0]

                    data = self._safe_to_tensor(batch['data'], dev, is_target=False)
                    target = self._safe_to_tensor(batch['target'], dev, is_target=True)

                    logits = self.network(data)
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    elif isinstance(logits, dict) and 'out' in logits:
                        logits = logits['out']

                    dices.append(self._quick_fg_dice_from_logits(logits, target))
                    tried += 1
        except StopIteration:
            pass
        except Exception as e:
            self.print_to_log_file(f"[probe train dice skipped: {e}]")
        finally:
            self.network.train()

        dices = [d for d in dices if not np.isnan(d)]
        return float(np.mean(dices)) if dices else float("nan")

    def on_epoch_end(self):
        ret = nnUNetTrainerV2_MedNeXt_M_kernel3.on_epoch_end(self)
        train_dice = self._estimate_train_dice_quick(max_batches=16)

        val_dice = float("nan")
        if hasattr(self, "all_val_eval_metrics") and len(self.all_val_eval_metrics) > 0:
            last_val = self.all_val_eval_metrics[-1]
            vals = None
            if isinstance(last_val, tuple) and len(last_val) == 2:
                vals = last_val[1]
            else:
                vals = last_val
            if vals is not None:
                vals = np.array(vals, dtype=float).ravel()
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    val_dice = float(vals.mean())

        n_steps = getattr(self, "num_batches_per_epoch", "N/A")
        ne = getattr(self, "max_num_epochs", "N/A")
        ep = getattr(self, "epoch", "N/A")

        self.print_to_log_file(
            f"[{self._opt_name}] Epoch {ep}/{ne} ({n_steps} iters) | "
            f"TrainDice={train_dice:.4f}  ValDice={val_dice:.4f}"
        )
        return ret