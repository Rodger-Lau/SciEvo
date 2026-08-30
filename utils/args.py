import argparse

def create_parser():
    parser = argparse.ArgumentParser()
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default="NYC")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    # Data parameters
    parser.add_argument('--num_nodes', type=int, default=206)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--tod_size', type=int, default=48)
    parser.add_argument('--data_root', type=str, default='data')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    parser.add_argument('--memory_coeff', '--rho_coeff', dest='memory_coeff', type=float, default=0.001, help='coefficient for rho = memory_coeff * (1 - mape)')
    parser.add_argument('--ltp_ltd_small_threshold', '--small_threshold', '--small_thereshold', dest='ltp_ltd_small_threshold', type=float, default=0.1, help='small activation threshold for LTD; LTP threshold is 1 - this value')
    # parser.add_argument('--milestones', nargs='+', type=int, default=[50, 120, 200])
    # parser.add_argument('--milestones', nargs='+', type=int, default=[50, 120])
    parser.add_argument('--steps', nargs='+', type=int, default=[30, 50])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epochs', type=int, default=100)
    #!
    # parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument('--threshold', type=float, default=0.000001)
    
    # Model arguments
    parser.add_argument('--obser_dim', type=int, default=3)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--d_obser', type=int, default=24)
    parser.add_argument('--d_tod', type=int, default=24)
    parser.add_argument('--d_dow', type=int, default=24)
    parser.add_argument('--d_ts', type=int, default=12)
    parser.add_argument('--d_s', type=int, default=12)
    parser.add_argument('--d_t', type=int, default=60)
    parser.add_argument('--d_p', type=int, default=72)
    parser.add_argument('--self_atten_dim', type=int, default=168)
    parser.add_argument('--cross_atten_dim', type=int, default=24)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Experiment control
    parser.add_argument('--experiment_name', type=str, default='full')
    parser.add_argument('--training_strategy', type=str, default='proposed', choices=['proposed', 'random_baseline'])
    parser.add_argument('--model_name', type=str, default='CMuST', help='model architecture to instantiate; register baseline models in main_ST.py')
    parser.add_argument('--baseline_input_dim', type=int, default=1, help='number of leading input features used by baseline models')
    parser.add_argument('--baseline_source_epochs', type=int, default=10)
    parser.add_argument('--baseline_target_epochs', type=int, default=50)
    parser.add_argument('--baseline_max_source_tasks', type=int, default=0, help='limit random baseline source tasks for smoke tests; 0 means use all')
    parser.add_argument('--num_random_runs', type=int, default=1)
    prediction_output_group = parser.add_mutually_exclusive_group()
    prediction_output_group.add_argument(
        '--save_sample_predictions',
        dest='disable_sample_predictions',
        action='store_false',
        help='save detailed sample prediction NPZ/CSV files (disabled by default)',
    )
    prediction_output_group.add_argument(
        '--disable_sample_predictions',
        dest='disable_sample_predictions',
        action='store_true',
        help='disable detailed sample prediction NPZ/CSV files (default)',
    )
    parser.set_defaults(disable_sample_predictions=True)
    parser.add_argument('--save_loss_curves', action='store_true', help='save per-epoch train/validation loss curves as CSV files')
    parser.add_argument('--log_cuda_memory', action='store_true', help='log current and peak CUDA allocated/reserved memory')
    parser.add_argument('--target_source_idx', type=int, default=0, help='source/domain index to use as the target domain')
    parser.add_argument('--target_task_idx', type=int, default=0, help='temporal task index to use as the target task')
    parser.add_argument('--task_per_dir', type=int, default=6, help='number of temporal domains per dataset split')
    parser.add_argument('--report_weekday_weekend_metrics', action='store_true')
    parser.add_argument('--report_hour_start', type=int, default=6)
    parser.add_argument('--report_hour_end', type=int, default=12)
    parser.add_argument(
        '--dropout_schedule',
        type=str,
        default='exp',
        choices=['exp', 'linear', 'inverse', 'log'],
        help='function used to map gradient magnitude to source-training dropout'
    )
    parser.add_argument(
        '--dropout_coeff_idx',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='coefficient set index for linear/inverse/log dropout schedules'
    )
    parser.add_argument(
        '--dropout_p0',
        type=float,
        default=0.1,
        help='p0 scale coefficient for the exponential dropout schedule'
    )
    parser.add_argument(
        '--source_order',
        type=str,
        default='normal',
        choices=['normal', 'domain_reverse', 'temporal_reverse', 'random', 'reverse'],
        help='source training order ablation for curriculum stages'
    )
    parser.add_argument('--train_source_idxs', nargs='*', type=int, default=None, help='source/domain indexes to use for source-domain training; defaults to all non-target domains')
    parser.add_argument('--ablation_modules', nargs='*', default=[])
    parser.add_argument('--few_shot_scale', type=float, default=0.3)
    parser.add_argument('--missing_steps', type=int, default=0, help='number of input timesteps to mask (from start)')
    parser.add_argument('--missing_node_ratio', type=float, default=0.0, help='fraction of spatial nodes to mask (0.0-1.0)')
    parser.add_argument('--missing_tail_steps', type=int, default=0, help='number of input timesteps to mask from the end')
    parser.add_argument('--missing_tail_node_ratio', type=float, default=0.0, help='fraction of spatial nodes to mask for tail-step missing (0.0-1.0)')
    parser.add_argument('--missing_tail_seed', type=int, default=0, help='random seed for selecting tail-step missing nodes')
    parser.add_argument('--missing_time_hours', type=float, default=0.0, help='length of a contiguous training time window to drop in hours')
    parser.add_argument('--missing_time_mode', type=str, default='random_contiguous', choices=['random_contiguous', 'fixed_contiguous'])
    parser.add_argument('--missing_calendar_pattern', type=str, default='none', choices=['none', 'daily_1h', 'weekly_1day', 'monthly_1week'])
    parser.add_argument('--missing_calendar_mode', type=str, default='random', choices=['random', 'fixed'])
    
    return parser
