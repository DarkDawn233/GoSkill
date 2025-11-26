import wandb
import itertools
import json5 as json
import time
from pathlib import Path
from collections import namedtuple
from copy import copy, deepcopy
import sys

from MTDT.model.prompt_dt import PromptDecisionTransformer
from MTDT.trainer.prompt_seq_trainer import PromptSequenceTrainer
from MTDT.envs.construct_envs import get_task_env_and_info
from MTDT.utils.process_data import process_total_data_mean, load_data_prompt, process_info
from MTDT.utils.params import get_default_args, get_args, recursive_dict_update, update_args

from MTDT.utils.others import set_seed, get_server_name

DATA_PATH = Path("/data1") # TODO

def experiment_mix_env(
        exp_prefix,
        variant,
):
    set_seed(variant['seed'])
    device = variant['device']
    log_to_wandb = variant['log_to_wandb']

    ######
    # construct train and test environments
    ######
    
    cur_dir = Path(__file__).resolve().parent
    env_config_save_path = cur_dir / 'config' / 'env'
    data_save_path = DATA_PATH
    model_save_path = cur_dir / 'model_saved'
    model_save_path.mkdir(exist_ok=True, parents=True)

    env_config_path_dict = {
        'metaworld-mt10': "metaworld/mt10.json",
        'metaworld-mt50': "metaworld/mt50.json",
        'metaworld-mt30': "metaworld/mt30.json",
        'metaworld-ml45': "metaworld/ml45.json",
    }
    
    task_config = env_config_save_path / env_config_path_dict[variant['env']]
    with open(task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    train_task_name_list, test_task_name_list = [], []
    for task in task_config.train_tasks:
        train_task_name_list.append(str(task))
    if variant['few_shot'] and hasattr(task_config, 'test_tasks'):
        for task in task_config.test_tasks:
            test_task_name_list.append(str(task))

    ######
    # process train and test datasets
    ######
    mode = variant.get('mode', 'normal')
    data_mode = variant['data_mode']

    # load training dataset
    trajectories_list, prompt_trajectories_list = load_data_prompt(variant['env'], train_task_name_list, data_save_path, data_mode, variant['preprocess_rtg'])
    if prompt_trajectories_list is None:
        prompt_trajectories_list = trajectories_list

    # training envs
    train_info, train_env_funcs = get_task_env_and_info(variant['env'], train_task_name_list,
                                                  trajectories_list, prompt_trajectories_list,
                                                  env_config_save_path, device, seed=variant['seed'])
    print(f'Train Env Info: {train_info} \n')

    # change to total train trajecotry
    if variant['average_state_mean']:
        train_traj_total = list(itertools.chain.from_iterable(trajectories_list))
        print("Total Train Trajectory Num: ", len(train_traj_total))
        total_state_mean, total_state_std= process_total_data_mean(train_traj_total, mode)
        variant['total_state_mean'] = total_state_mean
        variant['total_state_std'] = total_state_std

    # process train info
    train_info = process_info(train_task_name_list, trajectories_list, train_info, mode, data_mode, variant['pct_traj'], variant)

    ######
    # construct dt model and trainer
    ######
    env_abbreviate = {
        'metaworld-mt10': "mt10",
        'metaworld-mt50': "mt50",
        'metaworld-mt30': "mt30",
        'metaworld-ml45': "ml45",
    }
    exp_prefix = get_server_name() + '-' + exp_prefix + '-' + env_abbreviate[variant['env']]
    group_name = f'{exp_prefix}-{data_mode}'
    date_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    exp_prefix = f'{group_name}-{variant["seed"]}-{date_time}'

    state_dim = variant['state_dim']
    act_dim = variant['act_dim']
    
    # construct model post fix
    model_post_fix = '_TRAIN_'+variant['data_mode']

    model_config = variant['model']
    model = PromptDecisionTransformer(
        src_state_dim=variant['src_state_dim'],
        state_dim=state_dim,
        act_dim=act_dim,
        task_num=len(train_task_name_list),
        no_rtg=variant['no_rtg'],
        config=model_config
    )
    model = model.to(device=device)

    trainer = PromptSequenceTrainer(
        model=model,
        config=model_config
    )
    trainer.get_prompt_batch_fn(
        trajectories_list, prompt_trajectories_list, train_info, train_task_name_list,
        variant['no_state_normalize']
    )

    if variant['evaluation']:
        model_load_dir = Path(variant['load_path'])
        model_load_iter = variant['load_iter']
        trainer.load_model(
            env_name=variant['env'], 
            postfix=model_post_fix+'_iter_'+str(model_load_iter), 
            folder=model_load_dir
        )
        train_eval_logs = trainer.eval_iteration_multienv(
            train_info, train_task_name_list, train_env_funcs, prompt_trajectories_list, variant,
            iter_num=model_load_iter, print_logs=True, group='eval', vec_env_num=variant['vec_env_num'])
        
        return
    
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='Prompt-DT' + '-' + variant['env'],
            config=variant
        )
    model_save_path = model_save_path / exp_prefix
    model_save_path.mkdir(exist_ok=True)

    # Save config in model_save_dir
    with open(model_save_path / 'config.json', 'w') as f:
        variant_copy = copy(variant)
        if 'total_state_mean' in variant_copy:
            variant_copy['total_state_mean'] = variant_copy['total_state_mean'].tolist()
            variant_copy['total_state_std'] = variant_copy['total_state_std'].tolist()
        json.dump(variant_copy, f, indent=4)

    global_step = 0
        
    for iter in range(model_config['max_iters']):
        global_step += 1

        print("Train Iter:", iter)
        outputs = trainer.pure_train_iteration_mix()

        if not variant['few_shot'] and ((iter + 1) % variant['model']['train_eval_interval'] == 0 or iter == 0):
            # evaluate train
            train_eval_logs = trainer.eval_iteration_multienv(
                train_info, train_task_name_list, train_env_funcs, prompt_trajectories_list, variant,
                iter_num=iter + 1, print_logs=True, group='train', vec_env_num=variant['vec_env_num'])
            outputs.update(train_eval_logs)

        if (iter + 1) % variant['model']['save_interval'] == 0 or iter == 0:
            trainer.save_model(
                env_name=variant['env'], 
                postfix=model_post_fix+'_iter_'+str(global_step), 
                folder=model_save_path)

        outputs.update({"global_step": global_step}) # set global step as iteration
        
        if log_to_wandb:
            wandb.log(outputs)
    
    if variant['few_shot']:
        assert len(test_task_name_list) > 0, "No test task for few-shot evaluation"

        few_data_mode = variant['few_data_mode']
        # load test dataset
        test_trajectories_list, test_prompt_trajectories_list = load_data_prompt(variant['env'], test_task_name_list, data_save_path, few_data_mode, variant['preprocess_rtg'])
        if test_prompt_trajectories_list is None:
            test_prompt_trajectories_list = test_trajectories_list

        # test envs
        test_info, test_env_funcs = get_task_env_and_info(variant['env'], test_task_name_list,
                                                    test_trajectories_list, test_prompt_trajectories_list,
                                                    env_config_save_path, device, seed=variant['seed'])
        print(f'Test Env Info: {test_info} \n')
        
        # process test info
        test_info = process_info(test_task_name_list, test_trajectories_list, test_info, mode, few_data_mode, variant['pct_traj'], variant)

        trainer.get_prompt_batch_fn(
            test_trajectories_list, test_prompt_trajectories_list, test_info, test_task_name_list,
            variant['no_state_normalize']
        )

        for iter in range(model_config['few_max_iters']):
            global_step += 1

            print("Few Show Iter:", iter)
            outputs = trainer.pure_train_iteration_mix()

            if ((iter + 1) % variant['model']['few_eval_interval'] == 0 or iter == 0):
                # evaluate train
                test_eval_logs = trainer.eval_iteration_multienv(
                    test_info, test_task_name_list, test_env_funcs, prompt_trajectories_list, variant,
                    iter_num=iter + 1, print_logs=True, group='few-shot', vec_env_num=variant['vec_env_num'])
                outputs.update(test_eval_logs)

            if (iter + 1) % variant['model']['few_save_interval'] == 0 or iter == 0:
                trainer.save_model(
                    env_name=variant['env'], 
                    postfix=model_post_fix+'_iter_'+str(global_step), 
                    folder=model_save_path)

            outputs.update({"global_step": global_step}) # set global step as iteration
            
            if log_to_wandb:
                wandb.log(outputs)

        
if __name__ == '__main__':
    params = deepcopy(sys.argv)

    args = get_default_args()
    env_args = get_args(params, "--env_name", "env")
    args = recursive_dict_update(args, env_args)
    alg_args = get_args(params, "--alg_name", "alg")
    args = recursive_dict_update(args, alg_args)
    args = update_args(args, params)

    experiment_mix_env(args['alg_name'], variant=args)