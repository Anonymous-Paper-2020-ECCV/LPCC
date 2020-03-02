import os
import sys
import fire
import time
import torch

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_path)
sys.path.append(os.path.join(base_path, 'third_party'))

from configparser import ConfigParser
from third_party import torchplus
from optimizer import optimizer_builder
from optimizer import lr_scheduler_builder
from data.guo_car import KittiDetectionCar, car_merge_batch
from data.kitti_depth import KittiDepth, depth_merge_batch
from models.lpcc_net import LPCC_Net


def train_single_dataset(train_loader, val_loader, train_cfg, net, optimizer, lr_scheduler, result_dir, display_step,
                         save_step=None):
    # get configuration
    total_step = int(train_cfg['total_step'])
    eval_step_list = eval(train_cfg['eval_step_list'])
    if save_step is None:
        save_step_list = eval_step_list
    else:
        save_step_list = save_step

    # initialization
    optimizer.zero_grad()
    current_step = 0

    # main loop
    while current_step < total_step:
        for example_train in train_loader:
            # start step timer
            t = time.time()
            torch.cuda.synchronize()

            # lr scheduler step
            lr_scheduler.step(current_step)

            # network forward
            loss_dict, loss_info, _ = net(example_train)

            # get loss & backward
            loss = loss_dict['loss']
            loss.backward()

            # clip grad & optimizer step
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

            # stop step timer
            torch.cuda.synchronize()
            step_time = time.time() - t

            # display
            if current_step % display_step == 0 and current_step > 0:
                print(f"@@@ step: {current_step} @@@ == loss: {loss}, step_time: {step_time}")
                print(loss_info)

            # save checkpoints
            save_flag = (isinstance(save_step_list, list) and current_step in save_step_list) or \
                        (isinstance(save_step_list, int) and current_step % save_step_list == 0 and current_step > 0)
            if save_flag:
                torchplus.train.save_models(result_dir, [net, optimizer], current_step)

            # evaluation
            eval_flag = (isinstance(eval_step_list, list) and current_step in eval_step_list) or \
                        (isinstance(eval_step_list, int) and current_step % eval_step_list == 0 and current_step > 0)
            if eval_flag:
                net.eval()
                eval_results = []
                for example_val in val_loader:
                    with torch.no_grad:
                        eval_out, _ = net(example_val)
                    eval_results.append(eval_out)
                net.train()

            current_step += 1
            if current_step >= total_step:
                torchplus.train.save_models(result_dir, [net, optimizer], current_step - 1)
                break


# # TODO
# def train_all_dataset(train_depth_loader, val_depth_loader, train_car_loader,
#                       val_car_loader, train_cfg, net, optimizer, lr_scheduler, model_dir):
#     def generator(dataloader):
#         for data in dataloader:
#             yield data
#
#     set_total_step = eval(train_cfg['total_step'])
#     eval_step_list = eval(train_cfg['eval_step_list'])
#     optimizer.zero_grad()
#     total_step = 0
#     if len(train_depth_loader) >= len(train_car_loader):
#         long_train_loader = train_depth_loader
#         short_train_loader = train_car_loader
#     else:
#         long_train_loader = train_car_loader
#         short_train_loader = train_depth_loader
#
#     if len(val_depth_loader) >= len(val_car_loader):
#         long_val_loader = val_depth_loader
#         short_val_loader = val_car_loader
#     else:
#         long_val_loader = val_car_loader
#         short_val_loader = val_depth_loader
#     train_generator = generator(short_train_loader)
#     val_generator = generator(short_val_loader)
#
#     while total_step < set_total_step:
#         for example_train_long in long_train_loader:
#             try:
#                 example_train_short = next(train_generator)
#             except StopIteration:
#                 train_generator = generator(short_train_loader)
#                 example_train_short = next(train_generator)
#             else:
#                 pass
#             lr_scheduler.step(total_step)
#             # {"rgb": rgb, "sparse": sparse, "gt": target, 'calib': calib}
#             output_dict = net(example_train_long, example_train_short)
#             loss = output_dict['loss']
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
#             optimizer.step()
#             optimizer.zero_grad()
#
#             eval_flag = False
#             if isinstance(eval_step_list, list):
#                 if total_step in eval_step_list:
#                     eval_flag = True
#             if isinstance(eval_step_list, int):
#                 if total_step % eval_step_list == 0:
#                     eval_flag = True
#             if eval_flag:
#                 eval_results = []
#                 for example_val_long in long_val_loader:
#                     try:
#                         example_val_short = next(val_generator)
#                     except StopIteration:
#                         val_generator = generator(short_val_loader)
#                         example_val_short = next(val_generator)
#                     else:
#                         pass
#                     eval_out = net(example_val_long, example_val_short)
#                     eval_results.append(eval_out)
#                 torchplus.train.save_models(model_dir, [net, optimizer], total_step)
#             total_step += 1


def train(config_path,
          result_dir,
          display_step=50,
          save_step=None,
          resume=False):
    """
    main entrance for training

    :param config_path: configuration file
    :param result_dir: directory for saving models and logs
    :param display_step: display logs every display steps
    :param save_step: save checkpoint every save steps, same as evaluation steps if set to None
    :param resume: try resuming training from checkpoints, if specified
    :return: None
    """
    # get configuration
    config = ConfigParser()
    config.read(config_path)

    optimizer_config = config['OPTIMIZER']
    lrs_config = config['LR-SCHEDULER']
    train_config = config["TRAIN"]
    model_config = config['MODEL']

    dataset_mode = train_config['dataset_mode']

    # model dir & resume check
    if os.path.exists(result_dir) and resume:
        raise NotImplementedError('not implemented yet')
    elif os.path.exists(result_dir) and not resume:
        raise Exception('result_dir exists, but resume=False')
    elif not os.path.exists(result_dir) and resume:
        raise Exception('result_dir dose not exist, but resume=True')
    else:
        os.makedirs(result_dir)

    # prepare network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LPCC_Net(model_config).to(device)

    # optimizer
    optimizer = optimizer_builder.build(optimizer_config, net)
    lr_scheduler = lr_scheduler_builder.build(lrs_config, optimizer)

    # datasets
    # train_depth_dataset = KittiDepth(config['DATA-KITTI-DEPTH'], 'train')
    # val_depth_dataset = KittiDepth(config['DATA-KITTI-DEPTH'], 'val')
    train_car_dataset = KittiDetectionCar(config['DATA-CAR'], 'train')
    val_car_dataset = KittiDetectionCar(config['DATA-CAR'], 'val')

    # data loaders
    # train_depth_loader = torch.utils.data.DataLoader(
    #     train_depth_dataset,
    #     batch_size=eval(train_config['batch_size']),
    #     shuffle=True,
    #     num_workers=eval(train_config['num_workers']),
    #     pin_memory=False,
    #     collate_fn=depth_merge_batch
    # )
    # val_depth_loader = torch.utils.data.DataLoader(
    #     val_depth_dataset,
    #     batch_size=eval(train_config['batch_size']),
    #     shuffle=False,
    #     num_workers=eval(train_config['num_workers']),
    #     pin_memory=False,
    #     collate_fn=depth_merge_batch
    # )

    train_car_loader = torch.utils.data.DataLoader(
        train_car_dataset,
        batch_size=eval(train_config['batch_size']),
        shuffle=True,
        num_workers=eval(train_config['num_workers']),
        pin_memory=False,
        collate_fn=car_merge_batch
    )
    val_car_loader = torch.utils.data.DataLoader(
        val_car_dataset,
        batch_size=eval(train_config['batch_size']),
        shuffle=False,
        num_workers=eval(train_config['num_workers']),
        pin_memory=False,
        collate_fn=car_merge_batch
    )

    ####################################################
    # start training
    ####################################################
    # if dataset_mode == 'depth':
    #     train_single_dataset(train_depth_loader, val_depth_loader, train_config, net, optimizer, lr_scheduler,
    #                          result_dir, display_step)
    if dataset_mode == 'car':
        train_single_dataset(train_car_loader, val_car_loader, train_config, net, optimizer, lr_scheduler, result_dir,
                             display_step, save_step)
    # if dataset_mode == 'all':
    #     train_all_dataset(train_depth_loader, val_depth_loader, train_car_loader,
    #                       val_car_loader, train_config, net, optimizer, lr_scheduler, result_dir, display_step)


if __name__ == '__main__':
    fire.Fire()
