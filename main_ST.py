import numpy as np
import os
import torch
import torch.nn as nn
import random
# import datetime
from datetime import datetime
import copy
import csv
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from utils.utils import *
from utils.dataloader import *
from utils.logging import *
from utils.args import *
from engine import *
from model.models import CMuST
from tools.gradient_compute import *
from tools.cluster_ST import *
from tqdm import tqdm
from model.DQN import *
def get_config():
    parser = create_parser()
    args = parser.parse_args()
    
    # now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = 'logs/BrainAI/{}/{}/'.format(args.dataset, args.experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)
    
    return args, logger, now

def build_forecasting_model(args, device):
    model_name = args.model_name.lower()
    if model_name == 'cmust':
        return CMuST(
            num_nodes=args.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            tod_size=args.tod_size,
            obser_dim=args.obser_dim,
            output_dim=args.output_dim,
            d_obser=args.d_obser,
            d_tod=args.d_tod,
            d_dow=args.d_dow,
            d_ts=args.d_ts,
            d_s=args.d_s,
            d_t=args.d_t,
            d_p=args.d_p,
            self_atten_dim=args.self_atten_dim,
            cross_atten_dim=args.cross_atten_dim,
            ffn_dim=args.ffn_dim,
            n_heads=args.n_heads,
            dropout=args.dropout,
            device=device
        )
    raise ValueError('main_ST.py keeps only the SciEvo main experiment and requires model_name=CMuST.')

def prepare_domain_prompt(model, args, task_dir, device):
    if not hasattr(model, 'prompt'):
        return
    if args.dataset == 'SD':
        model.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.input_len, args.num_nodes, args.d_p)))
    else:
        load_prompt_weights(model, os.path.join(task_dir, "prompt.pth"), device_name=str(device))

if __name__ == "__main__":
    
    args, logger, now = get_config()
    try:
        do_write = False
        cross_temporal = False
        cross_source = True
        assert (cross_source != cross_temporal)
        logger.info(f'cross temporal: {cross_temporal}, cross source: {cross_source}')
        # is random seed
        if args.seed == 0:
            seed = random.randint(1, 1000)
        else:
            seed = args.seed
            
        set_seed(seed)
        # device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(args.gpu)
        if args.training_strategy != 'proposed':
            raise ValueError('main_ST.py keeps only the SciEvo main experiment; set training_strategy=proposed.')
        if args.model_name.lower() != 'cmust':
            raise ValueError('main_ST.py keeps only the SciEvo main experiment and requires model_name=CMuST.')
        
        dataset_dir = os.path.join(args.data_root, args.dataset)
        logger.info(f"ARGS num_nodes before model = {args.num_nodes}")
        # built model
        model = build_forecasting_model(args, device)
        logger.info(f"MODEL num_nodes = {getattr(model, 'num_nodes', getattr(model, 'num_node', args.num_nodes))}")
        
        # load dataset
        dataset_list = []
        
        if args.dataset == "NYC":
            dataset_list = ['CROWDIN', 'CROWDOUT', 'TAXIDROP', 'TAXIPICK']
        elif args.dataset == "CHI":
            dataset_list = ['RISK', 'TAXIPICK', 'TAXIDROP']
        elif args.dataset == 'SIP':
            dataset_list = ['FLOW', 'SPEED']
        else:
            dataset_list = ['SD']
            
        task_dirs = [os.path.join(dataset_dir, item) for item in dataset_list]
        logger.info(f"Load data {task_dirs}")
        new_i = args.target_source_idx # source domain
        new_j = args.target_task_idx # temporal task
        if new_i < 0 or new_i >= len(dataset_list):
            raise ValueError(f'target_source_idx={new_i} out of range for {args.dataset}: 0..{len(dataset_list)-1}')
        if args.train_source_idxs is None:
            train_source_idxs = [i for i in range(len(dataset_list)) if i != new_i]
        else:
            train_source_idxs = args.train_source_idxs
        if not train_source_idxs:
            raise ValueError('train_source_idxs must contain at least one source/domain index')
        invalid_train_source_idxs = [i for i in train_source_idxs if i < 0 or i >= len(dataset_list)]
        if invalid_train_source_idxs:
            raise ValueError(f'train_source_idxs contains out-of-range indexes {invalid_train_source_idxs} for {args.dataset}: 0..{len(dataset_list)-1}')
        if new_i in train_source_idxs:
            raise ValueError(f'train_source_idxs must not include target_source_idx={new_i} ({dataset_list[new_i]})')
        train_source_idx_set = set(train_source_idxs)
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        lengths = {}  # 数据规模
        cur_stds = []  # 标准差
        cur_means = []  # 均值
        scalers=[]
        few_shot_scale = args.few_shot_scale
        # Main experiment only: keep all SciEvo components enabled.
        state_encoding_mode = 'full'
        hierarchy = True
        logger.info(f'few shot scale: {few_shot_scale}')
        logger.info(f'experiment name: {args.experiment_name}')
        logger.info(f'evaluate domain: {dataset_list[new_i]}')
        logger.info(f'train source domains: {[dataset_list[i] for i in train_source_idxs]}')
        logger.info(
            f'dropout_schedule: {args.dropout_schedule}, '
            f'dropout_coeff_idx: {args.dropout_coeff_idx}, dropout_p0: {args.dropout_p0}'
        )
        logger.info(f'source_order: {args.source_order}')
        logger.info(
            f'tuning hyperparams: lr={args.lr}, memory_coeff={args.memory_coeff}, '
            f'ltp_ltd_small_threshold={args.ltp_ltd_small_threshold}, '
            f'ltp_ltd_big_threshold={1.0 - args.ltp_ltd_small_threshold}'
        )

        def ordered_domain_indices():
            domain_indices = list(range(len(dataset_list)))
            if args.source_order in {'domain_reverse', 'reverse'}:
                domain_indices.reverse()
            elif args.source_order == 'random':
                random.shuffle(domain_indices)
            return domain_indices

        def ordered_task_indices(domain_index):
            task_indices = list(range(len(train_loaders[domain_index])))
            if args.source_order in {'temporal_reverse', 'reverse'}:
                task_indices.reverse()
            elif args.source_order == 'random':
                random.shuffle(task_indices)
            return task_indices

        def apply_source_order(pairs):
            if args.source_order == 'domain_reverse':
                return sorted(pairs, key=lambda pair: (-pair[0], pair[1]))
            if args.source_order == 'temporal_reverse':
                return sorted(pairs, key=lambda pair: (pair[0], -pair[1]))
            if args.source_order == 'reverse':
                return list(reversed(pairs))
            if args.source_order == 'random':
                shuffled_pairs = list(pairs)
                random.shuffle(shuffled_pairs)
                return shuffled_pairs
            return pairs

        for i in range(len(dataset_list)):
            train_loaders[i] = {}
            val_loaders[i] = {}
            test_loaders[i] = {}
            lengths[i] = {}
            # stds[i] = {}
            # means[i] = {}
        for i in range(len(dataset_list)):
            # dataloaders, scaler = get_dataloaders_scaler(task_dir, args.batch_size, logger)
            if i != new_i:
                dataloaders_list, scalers_list, mean_list, std_list = get_dataloaders_scaler_and_split_task(
                    task_dirs[i],
                    args.batch_size,
                    args.task_per_dir,
                    logger,
                    missing_time_hours=args.missing_time_hours,
                    missing_time_mode=args.missing_time_mode,
                    missing_time_seed=seed,
                    missing_calendar_pattern=args.missing_calendar_pattern,
                    missing_calendar_mode=args.missing_calendar_mode,
                    missing_node_ratio=args.missing_node_ratio,
                    output_len=args.output_len,
                    missing_tail_steps=args.missing_tail_steps,
                    missing_tail_node_ratio=args.missing_tail_node_ratio,
                    missing_tail_seed=args.missing_tail_seed,
                )
            else:
                dataloaders_list, scalers_list, mean_list, std_list = get_dataloaders_scaler_and_split_task_few_shot(
                    task_dirs[i],
                    args.batch_size,
                    args.task_per_dir,
                    logger,
                    few_shot_scale,
                    missing_time_hours=args.missing_time_hours,
                    missing_time_mode=args.missing_time_mode,
                    missing_time_seed=seed,
                    missing_calendar_pattern=args.missing_calendar_pattern,
                    missing_calendar_mode=args.missing_calendar_mode,
                    missing_node_ratio=args.missing_node_ratio,
                    output_len=args.output_len,
                    missing_tail_steps=args.missing_tail_steps,
                    missing_tail_node_ratio=args.missing_tail_node_ratio,
                    missing_tail_seed=args.missing_tail_seed,
                )
            scalers.append(scalers_list)  # scalar与dataloader一一对应
            cur_stds.append(std_list)
            cur_means.append(mean_list)
            
            for j in range(len(dataloaders_list)):
                train_loaders[i][j]=(dataloaders_list[j]['train'])
                val_loaders[i][j]=(dataloaders_list[j]['val'])
                test_loaders[i][j]=(dataloaders_list[j]['test'])
                lengths[i][j] = len(train_loaders[i][j])
                # stds[i][j] = scalers[i][j].std
                # means[i][j] = scalers[i][j].mean
                logger.info(f'cur_means:{cur_means[i][j]}, cur_std:{cur_stds[i][j]}')
                
            # scalers.append(scalers_list)  # scalar与dataloader一一对应
        if new_j < 0 or new_j >= len(train_loaders[new_i]):
            raise ValueError(f'target_task_idx={new_j} out of range for target source {dataset_list[new_i]}: 0..{len(train_loaders[new_i])-1}')
        num_tasks = len(train_loaders)
        data_dim = 11
        # model structure information
        total_params = sum(param.nelement() for param in model.parameters())
        logger.info(f'The number of parameters: {total_params}')
        
        # Rolling Adaption
        threshold = args.threshold
        
        # ========================== 设置一些超参数 ==========================
        first_stage_num_epochs = 10
        num_new_loader_epochs = 100
        n_clusters = 2
        add_idx = [ (1,0), (0,2), (1,2)]
        # ========================== 取出最后一个 new_domain 用于最终训练/测试 ==========================
        new_train_loader = train_loaders[new_i][new_j]  # 留出最后一个数据集作为new_domain
        new_val_loader = val_loaders[new_i][new_j]
        new_test_loader = test_loaders[new_i][new_j]
        new_scaler = scalers[new_i][new_j]

        new_dataloader = {
            'train_loader': new_train_loader,
            'val_loader': new_val_loader, 
            'test_loader': new_test_loader
        }
        # ========================== 收集所有 tasks (i,j) 除了最后一个 new_domain ==========================
        tasks = []
        for i in range(len(train_loaders)):
            for j in range(len(train_loaders[i])):
                # 跳过最后一个(i,j) - 也就是 new_domain
                if cross_source:
                    if i == new_i:
                        continue
                    if i not in train_source_idx_set:
                        continue
                if cross_temporal:
                    if j == new_j:
                        continue
                tasks.append((i, j))
        
        # ========================== 定义用于保存 3 次实验结果的列表 ==========================
        all_mae_results = []
        all_rmse_results = []
        all_mape_results = []

        # ========================== 做 3 次随机顺序的实验 ==========================
        #!
        # num_random_runs = 3
        num_random_runs = 1
        
        curriculum_num_epochs = 10
        csv_dir = f'csv_files/BrainAI/{args.dataset}/{args.experiment_name}'
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)

        for run_idx in range(num_random_runs):

            # 准备CSV
            csv_file = f'{csv_dir}/training_log_{run_idx}.csv'
            param_names = [name for name, _ in model.named_parameters()]
            if do_write:
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'loss'] + [f'grad_norm/{name}' for name in param_names])


            common_layer_changes = [[0 for _ in range(len(train_loaders[0]))] for _ in range(len(train_loaders))]
            common_shift_changes = [[0 for _ in range(len(train_loaders[0]))] for _ in range(len(train_loaders))]
            sum_gradients = []
            cat_gradients = []
            logger.info(f"======================== 开始第 {run_idx+1}/{num_random_runs} 次随机训练顺序实验 ========================")
            for dataset_index in ordered_domain_indices():
                while len(sum_gradients) <= dataset_index:
                    sum_gradients.append([])
                    cat_gradients.append([])
                if cross_source and dataset_index not in train_source_idx_set:
                    logger.info(f"Skip curriculum learning for source domain {dataset_list[dataset_index]} (not selected for training)")
                    continue
                sum_gradients[dataset_index] = [None for _ in range(len(train_loaders[dataset_index]))]
                cat_gradients[dataset_index] = [None for _ in range(len(train_loaders[dataset_index]))]
                for dataset_sample_index in ordered_task_indices(dataset_index):   
                    logger.info(f"======================== Curriculum learning for task {dataset_index}, {dataset_sample_index} ========================")
                    
                    # ========================== 对 tasks 列表做一次随机打乱 ==========================
                    # np.random.shuffle(tasks)
                    
                    # train task 1
                    # dataset_index, dataset_sample_index = tasks[0]
                    
                    logger.info(f'Train for [task {dataset_sample_index} (数据集: {dataset_list[dataset_index]})]...')
                    
                    # dataloader
                    train_loader = train_loaders[dataset_index][dataset_sample_index]
                    val_loader   = val_loaders[dataset_index][dataset_sample_index]
                    test_loader  = test_loaders[dataset_index][dataset_sample_index]
                    scaler       = scalers[dataset_index][dataset_sample_index]
                    # for inputs, labels in train_loader:
                    #     logger.info(f'inputs shape:{inputs.shape}') # Arguments: (torch.Size([16, 12, 206, 11])
                    #     logger.info(f'labels shape:{labels.shape}') # Arguments: (torch.Size([16, 12, 206, 1])
                    #     exit(0)


                    # load prompt
                    if args.dataset == 'SD':
                        model.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.input_len, args.num_nodes, args.d_p)))
                    else:
                        load_prompt_weights(model, os.path.join(task_dirs[dataset_index], f"prompt.pth"))
                    
                    # set train
                    criterion = MaskedMAELoss()
                    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
                    scheduler = MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
                    # start train
                    model, _ = train(model, device, train_loader, val_loader, scaler, 
                                optimizer, scheduler, criterion, max_epochs=curriculum_num_epochs, patience=args.patience, 
                                logger=logger,
                                loss_curve_name=f'run{run_idx}_curriculum_{dataset_list[dataset_index]}_task{dataset_sample_index}')
                    sum_gradient, cat_gradient = compute_gradient(model,val_loader,device,criterion,scaler)
                    logger.info(f"sum gradient: {sum_gradient}, cat_grad_len: {len(cat_gradient)}")
                    sum_gradients[dataset_index][dataset_sample_index] = sum_gradient  # 存储梯度平方和
                    cat_gradients[dataset_index][dataset_sample_index] = cat_gradient  # 存储梯度梯度拼接向量，以便于求cos
            cluster_members = clustering_2d_list(n_clusters=n_clusters, tensor_2d_list=cat_gradients, algorithm='kmeans')
            logger.info(f'cluster members: {cluster_members}')
            cluster_grad = get_cluster_average(cluster_members=cluster_members, cat_grad_dict=cat_gradients)
            logger.info(f'cluster grad: {cluster_grad}')
            #min_grad, min_i,min_j = get_min_gradient(sum_gradients=cluster_grad, new_i=-1, new_j=-1)
            sum_gradients_cluster = [sum(cluster_grad[i]) for i in range(len(cluster_grad))]
            sorted_clusters_gradients_list = sort_by_original_gradient(sum_gradients=sum_gradients_cluster)
            # sum_gradients_cluster=[]
            # for i in range(len(cluster_grad)):
            #     sum_gradients_cluster.append([sum(cluster_grad[j]) for j in range(len(cluster_grad[i]))])
            logger.info(f'sorted_clusters_gradients_list: {sorted_clusters_gradients_list}')
            # exit(0)
            # 开始排序
            sorted_sum_gradients = []
            sorted_cat_gradients = []
            #!
            common_stage_num_epochs = 50
            # common_stage_num_epochs = 30

            common_learning_rate = args.lr
            weight_decay_common = 0.01
            common_model = CMuST(
                num_nodes=args.num_nodes,
                input_len=args.input_len,
                output_len=args.output_len,
                tod_size=args.tod_size,
                obser_dim=args.obser_dim,
                output_dim=args.output_dim,
                d_obser=args.d_obser,
                d_tod=args.d_tod,
                d_dow=args.d_dow,
                d_ts=args.d_ts,
                d_s=args.d_s,
                d_t=args.d_t,
                d_p=args.d_p,
                self_atten_dim=args.self_atten_dim,
                cross_atten_dim=args.cross_atten_dim,
                ffn_dim=args.ffn_dim,
                n_heads=args.n_heads,
                dropout=args.dropout,
                device=device
            )
            # total_params = sum(p.numel() for p in common_model.parameters()) # 计算模型参数总数
            # logger.info(f'total params:{total_params}')
            state_dim = 16
            # action_scale_map = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # Action 0-8
            action_scale_map = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] # Action 0-8
            # state_encoder = nn.Linear(total_params, data_dim)
            # state_encoder = StateEncoder(grad_dim=total_params, data_dim=data_dim, state_dim=state_dim).to(device)
            dqn_agent = DQNAgent(
                grad_dim=total_params,
                encoding_dim=data_dim,
                data_dim=data_dim,
                state_dim=state_dim,
                action_dim=len(action_scale_map),
                device=device,
                logger=logger,
                state_encoding_mode=state_encoding_mode,
            ).to(device)
            # rl_optimizer = optim.Adam(list(state_encoder.parameters()) + list(dqn_agent.q_net.parameters()), lr=0.001)
            rl_optimizer = optim.Adam(dqn_agent.q_net.parameters(), lr=0.001)
            H_matrix = {name: 0.0 for name, p in model.named_parameters()}
            logger.info("Start Training with DQN Layer Freezing...")
            #!
            num_circles = 3
            # num_circles = 1

            if args.dataset == 'SD':
                common_model.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.input_len, args.num_nodes, args.d_p)))
            else:
                load_prompt_weights(common_model, os.path.join(task_dirs[train_source_idxs[-1]], f"prompt.pth"))
            mape = 1
            p0 = args.dropout_p0
            # first_step_num_epochs = 50
            logger.info('=================================First step starts=========================================')
            if hierarchy:
                for circle_index in range(1):
                    for cluster_index in sorted_clusters_gradients_list:
                        #print(cluster_index)
                        cluster = cluster_members[cluster_index]  # cluster中含有domain
                        temp_gradients_list = []
                        for i in range(len(cluster)):
                            temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
                        temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
                        sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                        sorted_gradients_list = apply_source_order(sorted_gradients_list)
                        logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                        d_max = sum_gradients[sorted_gradients_list[-1][0]][sorted_gradients_list[-1][1]]
                        schedule_d_max = d_max
                        if args.dropout_schedule != 'exp':
                            schedule_d_max = max(
                                float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
                                for value in sum_gradients_cluster
                            )

                        # 乱序
                        # np.random.shuffle(sorted_gradients_list)

                        for i_j_couple in sorted_gradients_list:
                            if mape >= 1 : mape = 1
                            i = i_j_couple[0]
                            j = i_j_couple[1]
                            grad_norms = {name: [] for name, param in common_model.named_parameters()}
                            sorted_sum_gradients.append(sum_gradients[i][j])
                            sorted_cat_gradients.append(cat_gradients[i][j])
                            if cross_source:
                                if i == new_i or i not in train_source_idx_set or (i,j) in add_idx:
                                    continue
                            if cross_temporal:
                                if j == new_j or (i,j) in add_idx:
                                    continue
                            logger.info(f'task {j} for {task_dirs[i]} start training...')
                            train_loader = train_loaders[i][j]
                            val_loader = val_loaders[i][j]
                            test_loader = test_loaders[i][j]
                            cur_mean = torch.from_numpy(cur_means[i][j]).float().to(device)
                            cur_std = torch.from_numpy(cur_stds[i][j]).float().to(device)
                            scaler = scalers[i][j]
                            #difference_value = difference_matrix[i][j]  # difference value越小越靠前，且值>=0
                            difference_value = sum_gradients[i][j]
                            #d_max = sorted_gradients_list[-1]
                            #if not (i == min_i and j == min_j):
                            # p_common = my_sigmoid(-difference_value,circle+1)
                            p_common = dropout_schedule(
                                args.dropout_schedule,
                                args.dropout_coeff_idx,
                                sum_gradients_cluster[cluster_index],
                                schedule_d_max,
                                p0=p0,
                            )
                            # weight_decay_common = my_sigmoid(lambda0,difference_value,d_max)
                            # p_common = my_reverse(p0,difference_value)
                            # weight_decay_common = 0.1*my_reverse(p0,difference_value)
                            #!
                            p_common = float(np.clip(p_common, 0.0, 0.9))
                            
                            common_model.dropout = p_common
                            # common_model.dropout = 0.1
                            common_model_optimizer = Adam(common_model.parameters(), lr=common_learning_rate,
                                                                weight_decay=weight_decay_common)
                            common_model_scheduler = MultiStepLR(common_model_optimizer, milestones=args.steps,gamma=args.gamma)
                            common_model.to(device)
                            common_model, _ = train(common_model, device, train_loader, val_loader, scaler, 
                                common_model_optimizer, common_model_scheduler, criterion, max_epochs=common_stage_num_epochs, patience=args.patience, 
                                logger=logger,
                                loss_curve_name=f'run{run_idx}_first_step_{dataset_list[i]}_task{j}')


            logger.info('======================================Second step starts=============================================')
            for circle_index in range(num_circles):
                for cluster_index in sorted_clusters_gradients_list:
                    #print(cluster_index)
                    cluster = cluster_members[cluster_index]  # cluster中含有domain
                    temp_gradients_list = []
                    for i in range(len(cluster)):
                        temp_gradients_list.append(sum_gradients[cluster[i][0]][cluster[i][1]])
                    temp_sorted_gradients_list = sort_by_original_gradient(sum_gradients=temp_gradients_list)
                    sorted_gradients_list = [cluster[index] for index in temp_sorted_gradients_list]
                    sorted_gradients_list = apply_source_order(sorted_gradients_list)
                    logger.info(f'sorted_gradients_list: {sorted_gradients_list}')
                    d_max = sum_gradients[sorted_gradients_list[-1][0]][sorted_gradients_list[-1][1]]

                    # 乱序
                    # np.random.shuffle(sorted_gradients_list)

                    for i_j_couple in sorted_gradients_list:
                            if mape >= 1 : mape = 1
                            i = i_j_couple[0]
                            j = i_j_couple[1]
                            grad_norms = {name: [] for name, param in common_model.named_parameters()}
                            sorted_sum_gradients.append(sum_gradients[i][j])
                            sorted_cat_gradients.append(cat_gradients[i][j])
                            if cross_source:
                                if i == new_i or i not in train_source_idx_set:
                                    continue
                            if cross_temporal:
                                if j == new_j:
                                    continue
                            logger.info(f'task {j} for {task_dirs[i]} start training...')
                            train_loader = train_loaders[i][j]
                            val_loader = val_loaders[i][j]
                            test_loader = test_loaders[i][j]
                            cur_mean = torch.from_numpy(cur_means[i][j]).float().to(device)
                            cur_std = torch.from_numpy(cur_stds[i][j]).float().to(device)
                            scaler = scalers[i][j]
                            #difference_value = difference_matrix[i][j]  # difference value越小越靠前，且值>=0
                            difference_value = sum_gradients[i][j]
                            #d_max = sorted_gradients_list[-1]
                            #if not (i == min_i and j == min_j):
                            #p_common = my_sigmoid(-difference_value,circle+1)
                            # p_common = my_sigmoid(p0, difference_value,d_max)
                            # weight_decay_common = my_sigmoid(lambda0,difference_value,d_max)
                            # p_common = my_reverse(p0,difference_value)
                            # weight_decay_common = 0.1*my_reverse(p0,difference_value)
                            # Full model keeps second-step dropout at 0.1.
                            common_model.dropout = 0.1
                            common_model_optimizer = Adam(common_model.parameters(), lr=common_learning_rate,
                                                                weight_decay=weight_decay_common)
                            common_model_scheduler = MultiStepLR(common_model_optimizer, milestones=args.steps,gamma=args.gamma)
                            common_model.to(device)
                        # for epoch in tqdm(range(common_stage_num_epochs)):
                            # sum_grad, cat_grad = compute_gradient(model=common_model, train_loader=train_loader, device=device,criterion=criterion,in_dim=in_dim,scaler=scaler)
                            # los, activate_frequency = train(scaler,train_loader,epoch,common_model_optimizer,criterion,common_model,device,in_dim,
                            #                                 last_param=last_param,last_out=last_output)
                            common_model, w_list, common_layer_changes[i][j], common_shift_changes[i][j], mape = train_RL(common_model, device, train_loader, val_loader, scaler, 
                                common_model_optimizer, common_model_scheduler, criterion, max_epochs=common_stage_num_epochs, patience=args.patience, 
                                cur_mean=cur_mean, cur_std=cur_std, action_scale_map=action_scale_map, H_matrix=H_matrix, rl_optimizer=rl_optimizer, dqn_agent=dqn_agent,
                                loss_curve_name=f'run{run_idx}_second_step_{dataset_list[i]}_task{j}')
                            logger.info(f'common_layer_changes on task {j} for {task_dirs[i]}: {common_layer_changes[i][j]}')
                            logger.info(f'common_shift_changes on task {j} for {task_dirs[i]}: {common_shift_changes[i][j]}')
                            # train_cat_gradients.append(cat_grad)
                            # train_sum_gradients.append(sum_grad)
                            # for name, param in common_model.named_parameters():
                            #     if param.grad is not None and 'weight' in name:
                            #         grad_norm = param.grad.norm().item()
                            #         grad_norms[name].append(grad_norm)
                            # if LTP and LTP_bool and saved:
                            #             torch.save(common_model,f'./saved_models/LTP_post_common_model_{args.dataset}.pth')
                            #             print('发生LTP，保存post_LTP')
                            #             LTP_bool = False
                            # if LTD and LTD_bool and saved:
                            #             torch.save(common_model,f'./saved_models/LTD_post_common_model_{args.dataset}.pth')
                            #             print('发生LTD，保存post_LTD')
                            #             LTD_bool = False
                            # if epoch % 2 == 0:
                            #     print('第',epoch, '轮激活频率为', activate_frequency)
                            #     if activate_frequency > 0.2:
                            #         #print('增强')
                            #         if LTP_bool and saved:
                            #             torch.save(common_model,f'./saved_models/LTP_pre_common_model_{args.dataset}.pth')
                            #             print('发生LTP，保存pre_LTP,此时weight_decay=',weight_decay_common*1.2)
                            #             LTP = True
                            #             LTP_bool = False
                            #         weight_decay_common *= 1.2
                            #         optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            #         if common_model.dropout*1.2<1:
                            #             common_model.dropout *= 1.2
                            #     elif activate_frequency<0.1:
                            #         #print('削弱')
                            #         if LTD_bool and saved:
                            #             torch.save(common_model,f'./saved_models/LTD_pre_common_model_{args.dataset}.pth')
                            #             print('发生LTD，保存pre_LTD,此时weight_decay=',weight_decay_common*0.5)
                            #             LTD = True
                            #             LTD_bool = False
                            #         weight_decay_common *=0.5
                            #         optimizer = torch.optim.Adam(common_model.parameters(), lr=common_learning_rate, weight_decay=weight_decay_common)
                            #         common_model.dropout*=0.5
                        #     common_train_loss.append(los)
                        #     #common_train_loss.append(train(scaler,train_loader,epoch,common_model_optimizer,criterion,common_model,device,in_dim))
                        # #if epoch % 10 == 0:
                        #     mae,rmse,mape, out_put=validate(scaler, val_loader, criterion, common_model, device, in_dim)
                        #     common_val_mae.append(mae)
                        #     common_val_rmse.append(rmse)
                        #     common_val_mape.append(mape)
                        # last_param = mape
                        # last_output = out_put
                        #draw_curve(grad_norms=grad_norms,save_dir=f'./gradient_visualization_{args.dataset}',key=[i,j],circle=circle,saved=True)
                        # if saved:
                        #     draw_curve(grad_norms=grad_norms,save_dir=f'./gradient_visualization_{args.dataset}',key=[i,j],circle=circle,saved=True)
                        #     torch.save(common_model,f'./saved_models/{args.dataset}/common_model_{i}_{j}_end.pth')
                        #     print(f'已保存{args.dataset}_{i}_{j}')
            if True:
                    #!
                    new_domain_stage_num_epochs = 200
                    # new_domain_stage_num_epochs = 50

                    # # test model
                    # test(model, device, test_loader, scaler, logger=logger)
                    
                    # # save prompt
                    # save_dir = 'checkpoints/BrainAI/{}/{}/'.format(args.dataset,now)
                    # if not os.path.exists(save_dir):
                    #     os.makedirs(save_dir)
                    
                    # # save prompt and model
                    # save_prompt_weights(model, os.path.join(save_dir, f"start_prompt.pth"))
                    
                    # # update weight list
                    # weight_histories = {name: [] for name, param in model.named_parameters()}
                    # for name, param in model.named_parameters():
                    #     weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
                    
                    # # train to task k
                    # for i, task in enumerate(tasks):
                    #     if i == 0:
                    #         continue
                        
                    #     # load task
                    #     dataset_index, dataset_sample_index = tasks[i]
                        
                    #     # dataloader
                    #     train_loader = train_loaders[dataset_index][dataset_sample_index]
                    #     val_loader   = val_loaders[dataset_index][dataset_sample_index]
                    #     test_loader  = test_loaders[dataset_index][dataset_sample_index]
                    #     scaler       = scalers[dataset_index][dataset_sample_index]
                        
                    #     # reset train
                    #     optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                    #                     lr=args.lr*0.01, weight_decay=args.weight_decay, eps=1e-8)
                    #     scheduler = MultiStepLR(optimizer, milestones=[500], gamma=args.gamma)
                        
                    #     # train task 2 to k
                    #     logger.info(f'Train for task {i+1} also [task {dataset_sample_index} (数据集: {dataset_list[dataset_index]})]...')
                        
                    #     # load prompt
                    #     if args.dataset == 'SD':
                    #         model.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.input_len, args.num_nodes, args.d_p)))
                    #     else:
                    #         load_prompt_weights(model, os.path.join(task_dirs[dataset_index], f"prompt.pth"))
                        
                    #     # start train
                    #     model, w_list = train(model, device, train_loader, val_loader, scaler, 
                    #                     optimizer, scheduler, criterion, max_epochs=10, patience=args.patience, logger=logger)
                    #     # append weight
                    #     for w in w_list:
                    #         for name, param in w.items():
                    #             weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
                                
                    #     # cal var & frozen
                    #     for name, param in model.named_parameters():
                    #         if param.requires_grad == True:
                    #             variances = calculate_variance(weight_histories[name])
                    #             if 'prompt' not in name and np.all(variances < threshold):
                    #                 param.requires_grad = False
                                    
                    #     # print frozen params
                    #     trainable_params = sum(param.nelement() for param in filter(lambda p: p.requires_grad, model.parameters()))
                    #     frozen_params = total_params - trainable_params
                    #     logger.info('Total/Frozen Parameters: {}/{}'.format(total_params, frozen_params))
                        
                    #     # test model
                    #     test(model, device, test_loader, scaler, logger=logger)
                        
                    #     # update weight list
                    #     weight_histories = {name: [] for name, param in model.named_parameters()}
                    #     for name, param in model.named_parameters():
                    #         weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())

                    # # train task 1
                    # # reset train
                    # optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), 
                    #                     lr=args.lr*0.01, weight_decay=args.weight_decay, eps=1e-8)
                    # scheduler = MultiStepLR(optimizer, milestones=[500], gamma=args.gamma)
                    
                    # # train task 1
                    # dataset_index, dataset_sample_index = tasks[0]
                    
                    # logger.info(f'Train for task 1 also [task {dataset_sample_index} (数据集: {tasks[dataset_index]})]...')
                    
                    # # dataloader
                    # train_loader = train_loaders[dataset_index][dataset_sample_index]
                    # val_loader   = val_loaders[dataset_index][dataset_sample_index]
                    # test_loader  = test_loaders[dataset_index][dataset_sample_index]
                    # scaler       = scalers[dataset_index][dataset_sample_index]
                    
                    # # load prompt
                    # load_prompt_weights(model, os.path.join(save_dir, f"start_prompt.pth"))
                    
                    # # start train
                    # model, w_list = train(model, device, train_loader, val_loader, scaler, 
                    #                 optimizer, scheduler, criterion, max_epochs=10, patience=args.patience, logger=logger)
                    
                    # # append weight
                    # for w in w_list:
                    #     for name, param in w.items():
                    #         weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
                            
                    # # cal var & frozen
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad == True:
                    #         variances = calculate_variance(weight_histories[name])
                    #         if 'prompt' not in name and np.all(variances < threshold):
                    #             param.requires_grad = False
                                
                    # # print frozen params
                    # trainable_params = sum(param.nelement() for param in filter(lambda p: p.requires_grad, model.parameters()))
                    # frozen_params = total_params - trainable_params
                    # logger.info('Total/Frozen Parameters: {}/{}'.format(total_params, frozen_params))
                    
                    # # test model
                    # test(model, device, test_loader, scaler, logger=logger)
                    
                    # ========================== 在 new_domain 上训练并测试 ==========================
                    
                    logger.info('======================== 在 new_domain 上开始训练并测试 ========================')
                    
                    # reset train
                    optimizer = Adam(filter(lambda p : p.requires_grad, common_model.parameters()), 
                                    lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
                    scheduler = MultiStepLR(optimizer, milestones=args.steps, gamma=args.gamma)
                    
                    # load prompt
                    # if args.dataset == 'SD':
                    #     model.prompt = nn.init.xavier_uniform_(nn.Parameter(torch.empty(args.input_len, args.num_nodes, args.d_p)))
                    # else:
                    #     load_prompt_weights(model, os.path.join(task_dirs[dataset_index], f"prompt.pth"))
                    
                    # start train
                    # model, _ = train(model, device, new_train_loader, new_val_loader, new_scaler, 
                    #                 optimizer, scheduler, criterion, args.max_epochs, args.patience, logger=logger)
                    target_cur_mean = torch.from_numpy(cur_means[new_i][new_j]).float().to(device)
                    target_cur_std = torch.from_numpy(cur_stds[new_i][new_j]).float().to(device)
                    common_model, new_w_liat, new_layer_changes, new_shift_changes, new_mape = train_RL(common_model, device, new_train_loader, new_val_loader, new_scaler, 
                            optimizer, scheduler, criterion, max_epochs=new_domain_stage_num_epochs, patience=args.patience, 
                            cur_mean=target_cur_mean, cur_std=target_cur_std, action_scale_map=action_scale_map, H_matrix=H_matrix, rl_optimizer=rl_optimizer, dqn_agent=dqn_agent,
                            loss_curve_name=f'run{run_idx}_target_{dataset_list[new_i]}_task{new_j}')
                    
                    # test model
                    prediction_dir = os.path.join(csv_dir, 'sample_predictions')
                    prediction_file = None if args.disable_sample_predictions else os.path.join(
                        prediction_dir,
                        f'{dataset_list[new_i]}_task{new_j}_run{run_idx}.npz'
                    )
                    test_mae, test_rmse, test_mape = test(
                        common_model,
                        device,
                        new_test_loader,
                        new_scaler,
                        logger=logger,
                        prediction_save_path=prediction_file,
                    )
                
                    # 保留四位小数，并作为浮点数存储
                    test_mae  = round(test_mae, 4)
                    test_rmse = round(test_rmse, 4)
                    test_mape = round(test_mape, 4)
                    
                    logger.info(f"****** 第 {run_idx+1} 次实验 (随机顺序) 的测试结果: MAE={test_mae}, RMSE={test_rmse}, MAPE={test_mape} ******")
                    
                    # 把当前 run 的结果保存到列表中，便于后续统计
                    all_mae_results.append(test_mae)
                    all_rmse_results.append(test_rmse)
                    all_mape_results.append(test_mape)

        # ========================== 统计 3 次实验的平均结果并输出 ==========================
        mean_mae = sum(all_mae_results) / len(all_mae_results)
        mean_rmse = sum(all_rmse_results) / len(all_rmse_results)
        mean_mape = sum(all_mape_results) / len(all_mape_results)

        result_file = os.path.join(csv_dir, 'summary.csv')
        with open(result_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['experiment_name', 'dataset', 'seed', 'mae_mean', 'rmse_mean', 'mape_mean', 'mae_all', 'rmse_all', 'mape_all'])
            writer.writerow([
                args.experiment_name,
                args.dataset,
                seed,
                round(mean_mae, 4),
                round(mean_rmse, 4),
                round(mean_mape, 4),
                all_mae_results,
                all_rmse_results,
                all_mape_results,
            ])

        logger.info("======================================================")
        logger.info(f"三次随机顺序实验的 MAE : {all_mae_results}, 平均值: {round(mean_mae, 4)}")
        logger.info(f"三次随机顺序实验的 RMSE: {all_rmse_results}, 平均值: {round(mean_rmse, 4)}")
        logger.info(f"三次随机顺序实验的 MAPE: {all_mape_results}, 平均值: {round(mean_mape, 4)}")
        logger.info("======================================================")
            
        # test model
        # test(model, device, test_loader, scaler, logger=logger)
    except:
        log_cuda_memory(logger=logger, label='exception_cuda_memory', force=True)
        logger.exception('exception')
