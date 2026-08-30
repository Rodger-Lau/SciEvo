import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.metrics import get_mae_rmse_mape
from utils.args import *
import os
import csv
from datetime import datetime, timedelta
parser = create_parser()
args = parser.parse_args()
dataset = args.dataset
writer_dir = f'Writer/BrainAI/{dataset}/{args.experiment_name}'
if not os.path.exists(writer_dir):
    os.makedirs(writer_dir)
# writer = SummaryWriter(writer_dir)
# global_step = 0
small_threshold = args.ltp_ltd_small_threshold
big_threshold = 1.0 - small_threshold


def _save_loss_curve(loss_curve_name, train_losses, val_losses, lr_values, best_epoch, logger=None):
    if not getattr(args, 'save_loss_curves', False) or loss_curve_name is None:
        return

    loss_curve_dir = f'csv_files/BrainAI/{args.dataset}/{args.experiment_name}/loss_curves'
    os.makedirs(loss_curve_dir, exist_ok=True)
    safe_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in loss_curve_name)
    loss_curve_path = os.path.join(loss_curve_dir, f'{safe_name}.csv')
    with open(loss_curve_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'best_epoch'])
        for epoch, (train_loss, val_loss, lr) in enumerate(zip(train_losses, val_losses, lr_values), start=1):
            writer.writerow([epoch, train_loss, val_loss, lr, best_epoch])
    if logger is not None:
        logger.info(f'Saved loss curve to {loss_curve_path}')


def log_cuda_memory(logger=None, label='cuda_memory', force=False):
    if not torch.cuda.is_available():
        return
    if not force and not getattr(args, 'log_cuda_memory', False):
        return

    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    message = (
        f'{label}: current_allocated={allocated:.2f}MiB, '
        f'current_reserved={reserved:.2f}MiB, '
        f'peak_allocated={max_allocated:.2f}MiB, '
        f'peak_reserved={max_reserved:.2f}MiB, '
        f'total={total:.2f}MiB'
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)

def forward_with_activate_freq(model, inputs):
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        return outputs
    return outputs, 0.0

def _state_to_device(state, device):
    return tuple(item.to(device, non_blocking=True) for item in state)

def _state_shape_summary(state):
    return tuple(tuple(item.shape) for item in state)

def train_epoch(model, device, dataloader, scaler, optimizer, scheduler, criterion):
    model.to(device)
    model.train()
    losses = [] 
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs, _ = forward_with_activate_freq(model, inputs)
        preds = scaler.inverse_transform(outputs)

        loss = criterion(preds, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = np.mean(losses)
    scheduler.step()       
    return avg_loss

def train_epoch_RL(model, device, dataloader, scaler, optimizer, scheduler, criterion, cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, rl_optimizer, logger, use_learning_memory_interaction=True, use_fine_grained_editing=True, fixed_action_idx=4, state=None, new_domain=False, mape=1, activate_freq=0, missing_steps=0, use_reinforced_editing=True, use_mgo=True, use_arcon=True):
    rho = args.memory_coeff * (1 - mape) if use_mgo else 0.0
    losses = [] 
    # 保存当前参数，用于计算H更新
    prev_params = {name: param.data.clone() for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad = True
    if not use_fine_grained_editing and use_reinforced_editing:
        model.train()
        total_freq = 0
        batch_num = 0
        for inputs, labels in dataloader:
            batch_num += 1
            inputs = inputs.to(device)
            if missing_steps > 0:
                # mask the first `missing_steps` timesteps in the input
                if inputs.shape[1] >= missing_steps:
                    inputs[:, :missing_steps, ...] = 0
            labels = labels.to(device)
            outputs, activate_freq = forward_with_activate_freq(model, inputs)
            total_freq += activate_freq
            preds = scaler.inverse_transform(outputs)
            loss = criterion(preds, labels)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = np.mean(losses)
        scheduler.step()
        if batch_num > 0:
            activate_freq = total_freq / batch_num
        return model, avg_loss, H_matrix, state, activate_freq

    if not use_reinforced_editing:
        if logger is not None:
            logger.info('ReinEDIT disabled: bypass DQN and layer freezing; train all parameters')
            logger.info('Activation frequency is monitored without changing a freeze ratio')

        # The target-domain optimizer may have been built while some parameters were
        # frozen. Re-enable every parameter and add any omitted ones to the optimizer.
        for param in model.parameters():
            param.requires_grad = True
        optimized_param_ids = {id(param) for group in optimizer.param_groups for param in group['params']}
        missing_params = [param for param in model.parameters() if id(param) not in optimized_param_ids]
        if missing_params:
            optimizer.add_param_group({'params': missing_params})
            if hasattr(scheduler, 'base_lrs'):
                scheduler.base_lrs.extend(group['lr'] for group in optimizer.param_groups[len(scheduler.base_lrs):])
            if logger is not None:
                logger.info('ReinEDIT disabled: restored %d omitted parameters to optimizer', len(missing_params))

        model.train()
        total_freq = 0
        batch_num = 0
        for inputs, labels in dataloader:
            batch_num += 1
            inputs = inputs.to(device)
            if missing_steps > 0 and inputs.shape[1] >= missing_steps:
                inputs[:, :missing_steps, ...] = 0
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs, activate_freq = forward_with_activate_freq(model, inputs)
            total_freq += activate_freq
            preds = scaler.inverse_transform(outputs)
            training_labels = (1 - rho) * labels + rho * preds.detach() if use_mgo else labels
            loss = criterion(preds, training_labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        activate_freq = total_freq / batch_num if batch_num > 0 else 0
        updates = calculate_layer_updates(model, prev_params)
        avg_loss = np.mean(losses)
        for name, update_val in updates.items():
            H_matrix[name] = 0.9 * H_matrix.get(name, 0.0) + 0.1 * update_val
        scheduler.step()
        if logger is not None:
            logger.info('ReinEDIT disabled: activate freq:%s, rho:%s', activate_freq, rho)
        return model, avg_loss, H_matrix, state, activate_freq

    state_encoding_mode = getattr(dqn_agent, 'state_encoding_mode', 'full')
    uses_gradient_state = state_encoding_mode != 'data_only'
    if use_learning_memory_interaction and uses_gradient_state:
        model.eval()
        optimizer.zero_grad(set_to_none=True)
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            if missing_steps > 0:
                if inputs.shape[1] >= missing_steps:
                    inputs[:, :missing_steps, ...] = 0
            labels = labels.to(device)

            outputs, _ = forward_with_activate_freq(model, inputs)
            preds = scaler.inverse_transform(outputs)

            loss = criterion(preds, labels)
            loss.backward()
            break
        grad_flat = get_flat_grad(model).detach().cpu()
        optimizer.zero_grad(set_to_none=True)
        del inputs, labels, outputs, preds, loss
    elif use_learning_memory_interaction:
        grad_flat = torch.empty(0)
        logger.info('DataSE disabled: gradient state is omitted; policy uses mean/std only')
    else:
        grad_flat = None

    if use_learning_memory_interaction:
        grad_input = grad_flat.unsqueeze(0)
        mean_input = cur_mean.detach().cpu().unsqueeze(0)
        std_input = cur_std.detach().cpu().unsqueeze(0)
        logger.info(f'grad_input shape:{grad_input.shape}, mean_input shape:{mean_input.shape}, std_input shape:{std_input.shape}')
        action_state_vec = _state_to_device((grad_input, mean_input, std_input), device)
        action_idx = dqn_agent.select_action(action_state_vec, new_domain)
        del action_state_vec
        current_state_vec = (grad_input.squeeze(0), mean_input.squeeze(0), std_input.squeeze(0))
        scale = action_scale_map[action_idx]
    else:
        action_idx = fixed_action_idx
        current_state_vec = state
        scale = action_scale_map[action_idx]
        logger.info(f'fixed freeze scale action idx: {action_idx}, scale: {scale}')

    if not new_domain:
        logger.info(f'activate freq:{activate_freq}')
        if use_arcon:
            if activate_freq > big_threshold:
                logger.info('高激活频率，LTP')
                scale *= 0.9
            elif activate_freq < small_threshold:
                logger.info('低激活频率，LTD')
                scale /= 0.9
        else:
            logger.info('ARCon disabled: activation monitored, freeze scale unchanged')
    logger.info(f'freeze scale:{scale}')
    sorted_layers = sorted(H_matrix.items(), key=lambda item: item[1])

    total_layers = len(sorted_layers)
    freeze_count = int(total_layers * scale)
    layers_to_freeze = [name for name, _ in sorted_layers[:freeze_count]]

    model.train()
    for param in model.parameters():
        param.requires_grad = True
    freeze_layers(model, layers_to_freeze)

    total_freq = 0
    batch_num = 0
    for inputs, labels in dataloader:
        batch_num += 1
        inputs = inputs.to(device)
        if missing_steps > 0:
            if inputs.shape[1] >= missing_steps:
                inputs[:, :missing_steps, ...] = 0
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs, activate_freq = forward_with_activate_freq(model, inputs)
        total_freq += activate_freq
        preds = scaler.inverse_transform(outputs)
        soft_labels = (1 - rho) * labels + rho * preds.detach() if use_mgo else labels
        loss = criterion(preds, soft_labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    activate_freq = total_freq / batch_num if batch_num > 0 else 0
    next_state_vec = current_state_vec
    updates = calculate_layer_updates(model, prev_params)
    avg_loss = np.mean(losses)
    for name, update_val in updates.items():
        if name not in layers_to_freeze:
            H_matrix[name] = 0.9 * H_matrix[name] + 0.1 * update_val
    scheduler.step()

    if use_learning_memory_interaction and not new_domain:
        if uses_gradient_state:
            model.eval()
            for param in model.parameters():
                param.requires_grad = True
            optimizer.zero_grad(set_to_none=True)
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                if missing_steps > 0:
                    if inputs.shape[1] >= missing_steps:
                        inputs[:, :missing_steps, ...] = 0
                labels = labels.to(device)

                outputs, _ = forward_with_activate_freq(model, inputs)
                preds = scaler.inverse_transform(outputs)

                loss = criterion(preds, labels)
                loss.backward()
                break
            grad_flat = get_flat_grad(model).detach().cpu()
            optimizer.zero_grad(set_to_none=True)
            del inputs, labels, outputs, preds, loss
            logger.info(f'grad_flat shape:{grad_flat.shape}')
        else:
            grad_flat = torch.empty(0)
            logger.info('DataSE disabled: next policy state uses mean/std only')
        next_state_vec = (grad_flat, cur_mean.detach().cpu(), cur_std.detach().cpu())
        reward = 1 / avg_loss
        dqn_agent.memory.append((current_state_vec, action_idx, reward, next_state_vec))
        logger.info(
            f'current_state_shapes:{_state_shape_summary(current_state_vec)}, '
            f'action_idx:{action_idx}, reward:{reward}, '
            f'next_state_shapes:{_state_shape_summary(next_state_vec)}'
        )
        dqn_agent.train_step(batch_size=args.batch_size, optimizer=rl_optimizer)

    return model, avg_loss, H_matrix, next_state_vec, activate_freq


def train_RL(model, device, train_loader, val_loader, scaler, optimizer, scheduler, criterion, 
          max_epochs, patience, cur_mean, cur_std, action_scale_map, H_matrix, rl_optimizer, dqn_agent, csv_file=None, logger=None, write=False, use_learning_memory_interaction=True, use_fine_grained_editing=True, fixed_action_idx=4, new_domain=False, mape=1, loss_curve_name=None, use_reinforced_editing=True, use_mgo=True, use_arcon=True):
    model = model.to(device)
    writer = SummaryWriter(writer_dir)
    train_losses = []
    val_losses = []
    lr_values = []
    w_list = []
    
    early_stopping = EarlyStopping(patience=patience, trace_func=logger.info)
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    rel_changes = {}
    shift_changes = {}
    scale = 0
    activate_freq = 0.5
    torch.autograd.set_detect_anomaly(True)
    state_vec = None
    if logger is not None:
        logger.info(
            'EvoPoG/CoML controls: reinforced_editing=%s, mgo=%s, arcon=%s',
            use_reinforced_editing, use_mgo, use_arcon,
        )
        if not use_mgo:
            logger.info('MGO disabled: rho is forced to 0 and real labels are used')
        if not use_arcon:
            logger.info('ARCon disabled: activation frequency remains monitored without scale adjustment')
    for epoch in range(max_epochs):
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if write:
                    writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
            else:
                grad_norms.append(0.0)
                # writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
        
        model, train_loss, H_matrix, state_vec, activate_freq = train_epoch_RL(model, device, train_loader, scaler, optimizer, scheduler, criterion, 
                 cur_mean, cur_std, dqn_agent, action_scale_map, H_matrix, rl_optimizer, logger=logger, use_learning_memory_interaction=use_learning_memory_interaction, use_fine_grained_editing=use_fine_grained_editing, fixed_action_idx=fixed_action_idx, state=state_vec, new_domain=new_domain, mape=mape, activate_freq=activate_freq, missing_steps=args.missing_steps, use_reinforced_editing=use_reinforced_editing, use_mgo=use_mgo, use_arcon=use_arcon)
        if write:
            writer.add_scalar('Loss/train_batch', train_loss, epoch)
            with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch, train_loss.item()] + grad_norms)
        train_losses.append(train_loss)
        
        parameters_copy = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        w_list.append(parameters_copy)

        _,_,_,val_loss = eval(model, device, val_loader, scaler, criterion)
        val_losses.append(val_loss)
        lr_values.append(scheduler.get_last_lr()[0])

        message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
        logger.info(message.format(epoch+1, train_loss, val_loss, lr_values[-1]))

        early_stopping(val_loss, model)
        if epoch == 5:
            for name, param in model.named_parameters():
                old = old_params[name]
                new = param.detach()
                
                # 计算新旧参数的 L2 范数
                old_norm = old.norm().item()
                new_norm = new.norm().item()
                
                # 计算参数更新量的 L2 范数
                delta = new - old
                delta_norm = delta.norm().item()
                
                # 计算相对变化率（添加小常数防止除零）
                rel_change = delta_norm / (old_norm + 1e-12)
                shift_changes[name] = rel_change
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {early_stopping.best_epoch}")
            break

    
    # load best
    model.load_state_dict(early_stopping.best_checkpoint)
    best_epoch = early_stopping.best_epoch
    for name, param in model.named_parameters():
            old = old_params[name]
            new = param.detach()
            
            # 计算新旧参数的 L2 范数
            old_norm = old.norm().item()
            new_norm = new.norm().item()
            
            # 计算参数更新量的 L2 范数
            delta = new - old
            delta_norm = delta.norm().item()
            
            # 计算相对变化率（添加小常数防止除零）
            rel_change = delta_norm / (old_norm + 1e-12)
            rel_changes[name] = rel_change
    # eval model
    train_mae, train_rmse, train_mape, _ = eval(model, device, train_loader, scaler, criterion)
    val_mae, val_rmse, val_mape, _ = eval(model, device, val_loader, scaler, criterion)
    
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_losses[best_epoch-1], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_losses[best_epoch-1], val_mae, val_rmse, val_mape))
    _save_loss_curve(loss_curve_name, train_losses, val_losses, lr_values, best_epoch, logger=logger)
    log_cuda_memory(logger=logger, label=f'train_RL_end:{loss_curve_name}')

    return model, w_list, rel_changes, shift_changes, val_mape


def train(model, device, train_loader, val_loader, scaler, optimizer, scheduler, criterion, 
          max_epochs, patience, csv_file=None, logger=None, write=False, loss_curve_name=None):
    model = model.to(device)
    writer = SummaryWriter(writer_dir)
    train_losses = []
    val_losses = []
    lr_values = []
    w_list = []
    
    early_stopping = EarlyStopping(patience=patience, trace_func=logger.info)
    old_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    for epoch in range(max_epochs):
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if write:
                    writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
            else:
                grad_norms.append(0.0)
                # writer.add_scalar(f'grad_norm/{name}', grad_norm, epoch)
        train_loss = train_epoch(model, device, train_loader, scaler, optimizer, scheduler, criterion)
        if write:
            writer.add_scalar('Loss/train_batch', train_loss, epoch)
            with open(csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([epoch, train_loss.item()] + grad_norms)
        train_losses.append(train_loss)
        
        parameters_copy = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        w_list.append(parameters_copy)

        _,_,_,val_loss = eval(model, device, val_loader, scaler, criterion)
        val_losses.append(val_loss)
        lr_values.append(scheduler.get_last_lr()[0])

        message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
        logger.info(message.format(epoch+1, train_loss, val_loss, lr_values[-1]))

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {early_stopping.best_epoch}")
            break

    
    # load best
    model.load_state_dict(early_stopping.best_checkpoint)
    best_epoch = early_stopping.best_epoch
    # eval model
    train_mae, train_rmse, train_mape, _ = eval(model, device, train_loader, scaler, criterion)
    val_mae, val_rmse, val_mape, _ = eval(model, device, val_loader, scaler, criterion)
    
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_losses[best_epoch-1], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_losses[best_epoch-1], val_mae, val_rmse, val_mape))
    _save_loss_curve(loss_curve_name, train_losses, val_losses, lr_values, best_epoch, logger=logger)
    log_cuda_memory(logger=logger, label=f'train_end:{loss_curve_name}')

    return model, w_list

@torch.no_grad()
def eval(model, device, dataloader, scaler, criterion):
    
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = forward_with_activate_freq(model, inputs)
        preds = scaler.inverse_transform(outputs)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)         # [all_samples,12,num_nodes,dims]
    all_labels = torch.cat(all_labels, dim=0)       # [all_samples,12,num_nodes,dims]
    
    mae, rmse, mape = get_mae_rmse_mape(all_preds, all_labels)
    loss = criterion(all_preds, all_labels)

    return mae, rmse, mape, loss.item()


def _restore_input_scale(inputs, scaler):
    inputs = inputs.clone()
    inputs[..., 0] = scaler.inverse_transform(inputs[..., 0])
    return inputs.numpy()


def _extract_lon_lat(inputs):
    coords = inputs[..., 3:5]
    valid = np.any(coords != 0, axis=-1)
    longitude = coords[:, -1, :, 0].copy()
    latitude = coords[:, -1, :, 1].copy()
    assigned = np.zeros_like(valid[:, 0, :], dtype=bool)

    for time_idx in range(coords.shape[1]):
        use_idx = valid[:, time_idx, :] & ~assigned
        longitude[use_idx] = coords[:, time_idx, :, 0][use_idx]
        latitude[use_idx] = coords[:, time_idx, :, 1][use_idx]
        assigned |= use_idx

    return longitude, latitude


def _timestamp_from_components(components):
    try:
        year, month, day, hour, minute, second = [int(round(float(value))) for value in components]
        if year <= 0 or month <= 0 or day <= 0:
            return None
        return datetime(year, month, day, hour, minute, second)
    except (TypeError, ValueError, OverflowError):
        return None


def _format_timestamp(timestamp):
    if timestamp is None:
        return ''
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')


def _extract_input_timestamps(inputs):
    num_samples, input_len, num_nodes, _ = inputs.shape
    timestamps = np.empty((num_samples, input_len, num_nodes), dtype=object)
    for sample_id in range(num_samples):
        for time_idx in range(input_len):
            for node_id in range(num_nodes):
                timestamps[sample_id, time_idx, node_id] = _timestamp_from_components(
                    inputs[sample_id, time_idx, node_id, 5:11]
                )
    return timestamps


def _infer_prediction_timestamps(input_timestamps, output_len):
    num_samples, input_len, num_nodes = input_timestamps.shape
    prediction_timestamps = np.empty((num_samples, output_len, num_nodes), dtype=object)
    fallback_delta = timedelta(days=1.0 / max(getattr(args, 'tod_size', 1), 1))

    for sample_id in range(num_samples):
        for node_id in range(num_nodes):
            valid_times = [
                input_timestamps[sample_id, time_idx, node_id]
                for time_idx in range(input_len)
                if input_timestamps[sample_id, time_idx, node_id] is not None
            ]
            if not valid_times and node_id != 0:
                valid_times = [
                    input_timestamps[sample_id, time_idx, 0]
                    for time_idx in range(input_len)
                    if input_timestamps[sample_id, time_idx, 0] is not None
                ]
            if not valid_times:
                for horizon in range(output_len):
                    prediction_timestamps[sample_id, horizon, node_id] = None
                continue

            deltas = [
                valid_times[idx] - valid_times[idx - 1]
                for idx in range(1, len(valid_times))
                if valid_times[idx] > valid_times[idx - 1]
            ]
            step_delta = deltas[-1] if deltas else fallback_delta
            last_time = valid_times[-1]
            for horizon in range(output_len):
                prediction_timestamps[sample_id, horizon, node_id] = last_time + step_delta * (horizon + 1)

    return prediction_timestamps


def _timestamp_array_to_strings(timestamps):
    formatter = np.vectorize(_format_timestamp, otypes=[object])
    return formatter(timestamps)


def _sample_weekday_from_input(inputs, sample_id):
    timestamp = _timestamp_from_components(inputs[sample_id, -1, 0, 5:11])
    if timestamp is not None:
        return timestamp.weekday()
    dow_value = int(round(float(inputs[sample_id, -1, 0, 2])))
    if 0 <= dow_value <= 6:
        return dow_value
    return None


def _write_weekday_weekend_metrics(inputs, preds, labels, csv_path, logger=None):
    if not getattr(args, 'report_weekday_weekend_metrics', False):
        return

    rows = []
    start_hour = args.report_hour_start
    end_hour = args.report_hour_end
    hours = inputs[:, -1, 0, 8].astype(int)
    in_window = (hours >= start_hour) & (hours < end_hour)

    split_masks = {
        'weekday': np.zeros(inputs.shape[0], dtype=bool),
        'weekend': np.zeros(inputs.shape[0], dtype=bool),
    }
    for sample_id in range(inputs.shape[0]):
        if not in_window[sample_id]:
            continue
        weekday = _sample_weekday_from_input(inputs, sample_id)
        if weekday is None:
            continue
        if weekday < 5:
            split_masks['weekday'][sample_id] = True
        else:
            split_masks['weekend'][sample_id] = True

    for split_name, mask in split_masks.items():
        sample_count = int(mask.sum())
        if sample_count == 0:
            rows.append([split_name, start_hour, end_hour, 0, '', '', ''])
            if logger is not None:
                logger.info(f'{split_name} {start_hour}-{end_hour} metrics skipped: no samples')
            continue
        tensor_mask = torch.as_tensor(mask, dtype=torch.bool)
        mae, rmse, mape = get_mae_rmse_mape(preds[tensor_mask], labels[tensor_mask])
        rows.append([split_name, start_hour, end_hour, sample_count, mae, rmse, mape])
        if logger is not None:
            logger.info(
                f'{split_name} {start_hour}-{end_hour} samples={sample_count}: '
                f'MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}'
            )

    if csv_path is None:
        return
    metrics_path = f'{os.path.splitext(csv_path)[0]}_weekday_weekend_metrics.csv'
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'hour_start', 'hour_end', 'sample_count', 'mae', 'rmse', 'mape'])
        writer.writerows(rows)
    if logger is not None:
        logger.info(f'Saved weekday/weekend metrics to {metrics_path}')


def _save_sample_predictions(inputs, preds, labels, scaler, save_path, logger=None):
    if save_path is None:
        return

    save_path = os.fspath(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    inputs_np = _restore_input_scale(inputs, scaler)
    preds_np = preds.numpy()
    labels_np = labels.numpy()
    longitude, latitude = _extract_lon_lat(inputs_np)
    input_timestamps = _extract_input_timestamps(inputs_np)
    prediction_timestamps = _infer_prediction_timestamps(input_timestamps, preds_np.shape[1])
    input_timestamp_strings = _timestamp_array_to_strings(input_timestamps)
    prediction_timestamp_strings = _timestamp_array_to_strings(prediction_timestamps)

    npz_path = save_path if save_path.endswith('.npz') else f'{save_path}.npz'
    csv_path = f'{os.path.splitext(npz_path)[0]}.csv'
    _write_weekday_weekend_metrics(inputs_np, preds, labels, csv_path, logger=logger)

    np.savez_compressed(
        npz_path,
        inputs=inputs_np,
        predictions=preds_np,
        truths=labels_np,
        longitude=longitude,
        latitude=latitude,
        input_timestamps=input_timestamp_strings,
        prediction_timestamps=prediction_timestamp_strings,
        feature_names=np.array([
            'value', 'time_of_day', 'day_of_week', 'longitude', 'latitude',
            'year', 'month', 'day', 'hour', 'minute', 'second'
        ]),
    )

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id',
            'horizon',
            'node_id',
            'target_dim',
            'prediction_timestamp',
            'prediction',
            'truth',
            'longitude',
            'latitude',
            'input_timestamps',
            'input_values',
        ])
        num_samples, output_len, num_nodes, output_dim = preds_np.shape
        for sample_id in range(num_samples):
            for horizon in range(output_len):
                for node_id in range(num_nodes):
                    input_timestamps_text = ';'.join(input_timestamp_strings[sample_id, :, node_id].tolist())
                    input_values_text = ';'.join(f'{value:.6g}' for value in inputs_np[sample_id, :, node_id, 0])
                    for target_dim in range(output_dim):
                        writer.writerow([
                            sample_id,
                            horizon + 1,
                            node_id,
                            target_dim,
                            prediction_timestamp_strings[sample_id, horizon, node_id],
                            preds_np[sample_id, horizon, node_id, target_dim],
                            labels_np[sample_id, horizon, node_id, target_dim],
                            longitude[sample_id, node_id],
                            latitude[sample_id, node_id],
                            input_timestamps_text,
                            input_values_text,
                        ])

    if logger is not None:
        logger.info(f'Saved sample predictions to {npz_path} and {csv_path}')


@torch.no_grad()
def test(model, device, test_loader, scaler, logger, prediction_save_path=None):
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_inputs = [] if prediction_save_path is not None or getattr(args, 'report_weekday_weekend_metrics', False) else None
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = forward_with_activate_freq(model, inputs)
        preds = scaler.inverse_transform(outputs)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        if all_inputs is not None:
            all_inputs.append(inputs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if all_inputs is not None:
        all_inputs = torch.cat(all_inputs, dim=0)
    
    test_maes = []
    test_mapes = []
    test_rmses = []
    
    output_len = all_labels.shape[1]
    for i in range(output_len):
        mae, rmse, mape = get_mae_rmse_mape(all_preds[:,i,...], all_labels[:,i,...])
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        logger.info(log.format(i + 1, mae, rmse, mape))
        test_maes.append(mae)
        test_rmses.append(rmse)
        test_mapes.append(mape)

    log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    logger.info(log.format(np.mean(test_maes), np.mean(test_rmses), np.mean(test_mapes)))
    _save_sample_predictions(all_inputs, all_preds, all_labels, scaler, prediction_save_path, logger=logger)
    if prediction_save_path is None and getattr(args, 'report_weekday_weekend_metrics', False):
        inputs_np = _restore_input_scale(all_inputs, scaler)
        _write_weekday_weekend_metrics(inputs_np, all_preds, all_labels, None, logger=logger)
    
    return np.mean(test_maes), np.mean(test_rmses), np.mean(test_mapes)
    
