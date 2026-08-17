from options.base_option import BaseOptions


class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')
        # Training masking-ratio distribution ablation (MMM r~U(alpha,1) vs default cosine)
        self.parser.add_argument('--train_mask_dist', default='cosine', choices=['cosine', 'uniform'],
                            help="training masking-ratio distribution: 'cosine' (deployed default) or 'uniform' = U(alpha,1)")
        self.parser.add_argument('--train_mask_alpha', type=float, default=0.0,
                            help='alpha for --train_mask_dist uniform (r ~ U(alpha,1))')
        self.parser.add_argument('--train_split', default='train160k.txt', help='training split filename under data_root')
        self.parser.add_argument('--val_split', default='val40k.txt', help='validation split filename under data_root')
        self.parser.add_argument('--stream_img_feat', action='store_true',
                            help='stream DINOv2 features from disk per-item instead of preloading all into RAM '
                                 '(needed for 160k towers / parallel runs; ~640GB/process otherwise)')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (iterations). Ignored if --milestone_fractions is set.")
        self.parser.add_argument('--milestone_fractions', default=None, nargs="+", type=float,
                            help="Fractions of total training (e.g. 0.75 0.92) to drop LR. Overrides --milestones. "
                                 "Resolved at runtime from max_epoch × iters/epoch, survives wrong iters/epoch estimates.")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument('--view_drop_prob', type=float, default=0.0, help='Per-view dropout (patch_clsprefix only): independently zero each of the 4 patch-feature views with this prob, on top of the all-or-nothing cond_drop. Makes partial-view conditioning in-distribution for few-view inference.')
        self.parser.add_argument('--coord_loss_weight', type=float, default=0.0, help='Weight for coordinate reconstruction loss (0=disabled, try 1.0 for image_patch)')
        self.parser.add_argument('--ce_loss_weight', type=float, default=1.0, help='Weight for CE loss. Set <1 (e.g. 0.1) for coord-loss-dominant training (v7+); CE then acts only as entropy/anchor.')
        self.parser.add_argument('--use_cls_prefix', action='store_true', help='Patch model only (v5): v4 cross-attn + 4 CLS-style prefix tokens (per-view mean-pooled patches) prepended to the motion sequence.')
        self.parser.add_argument('--val_decode_mode', type=str, default='windowed', choices=['windowed', 'whole', 'both'], help="Decode mode for periodic val-IoU eval. 'windowed' (RVQ default, w=64), 'whole' (IBQ deployment decode), 'both' (log both; net_best_iou tracks whole).")
        self.parser.add_argument('--grad_probe_every', type=int, default=0, help='Log per-loss gradient norms every N iters (0=disabled). Adds ~1.5%% overhead at probe step. Use ~200 for diagnostic.')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")

        self.parser.add_argument('--deterministic_crop', action="store_true", help='Disable random crop augmentation in Image2TowerDataset: always the full sequence from index 0 (fixes tower sequence-end weakness; random crops are a human-motion inheritance)')
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--reset_optimizer', action="store_true", help='Load model weights only (no optimizer/scheduler state); use with --is_continue for warm restart at a new LR')
        self.parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes (use 0 to debug hanging DataLoader)')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=10, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')


        self.is_train = True
