import os
import shutil
import random
import time

datasets = '20ng,R8,R52,ohsumed,mr'.split(',')
models = 'bert-base-uncased,roberta-base'.split(',')
unlearns = 'raw,retrain,FF,GA,IU,FT'.split(',')
metrics = 'RA,UA,TA'.split(',')

ext_args = {
    "retrain": " --unlearn_epochs 60",
    "IU": " --alpha 5",
    "FF": " --alpha 2e-7",
    "GA": " --unlearn_lr 5e-6",
}

# Feb. 11, 2023 version
def run_commands(gpus, commands, suffix, call=False, shuffle=True, delay=0.5, ext_command=""):
    command_dir = os.path.join("commands", suffix)
    if len(commands) == 0:
        return
    if os.path.exists(command_dir):
        shutil.rmtree(command_dir)
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)
    os.makedirs(command_dir, exist_ok=True)

    stop_path = os.path.join('commands', 'stop_{}.sh'.format(suffix))
    with open(stop_path, 'w') as fout:
        print("kill $(ps aux|grep 'bash " + command_dir +
              "'|awk '{print $2}')", file=fout)

    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if len(i_commands) == 0:
            continue
        prefix = "CUDA_VISIBLE_DEVICES={} ".format(gpu)
        ext_command_i = ext_command.format(i=i)

        sh_path = os.path.join(command_dir, "run{}.sh".format(i))
        fout = open(sh_path, 'w')
        for com in i_commands:
            print(prefix + com + ext_command_i, file=fout)
        fout.close()
        if call:
            os.system("bash {}&".format(sh_path))
            time.sleep(delay)


def gen_commands_preprocess():
    commands = []
    for dataset in datasets:
        command = f"python build_graph.py {dataset}"
        commands.append(command)
    return commands

def gen_commands_finetune():
    commands = []
    for dataset in datasets:
        for model in models:
            command = f"python finetune_bert.py --dataset {dataset} --bert_init {model}"
            commands.append(command)
    return commands

def gen_commands_unlearn(rerun=False):
    commands = []
    for dataset in datasets:
        for model in models:
            for unlearn in unlearns:
                command = f"python unlearn_bert.py --dataset {dataset} --unlearn-method {unlearn} --bert_init {model}"
                if rerun:
                    command += " --rerun"
                if unlearn in ext_args:
                    command += ext_args[unlearn]
                commands.append(command)
    return commands

def gen_grid_search():
    commands = []
    alphas = list(range(20))

    for dataset in datasets:
        for model in models:
            for alpha in alphas:
                unlearn = "IU"
                command = f"python unlearn_bert.py --dataset {dataset} --unlearn-method {unlearn} --bert_init {model} --alpha {alpha}"
                command += f" --save-dir grid_results/{dataset}/{model}/{unlearn}/alpha_{alpha}"
                if unlearn in ext_args:
                    command += ext_args[unlearn]
                commands.append(command)
    return commands

if __name__ == "__main__":
    # commands = gen_commands_unlearn(False)
    # run_commands([0,1,2], commands, "unlearn", call=False, shuffle=False, delay=1)
    commands = gen_grid_search()
    run_commands([0,1,2,3] * 3, commands, "grid", call=True, shuffle=False, delay=1)