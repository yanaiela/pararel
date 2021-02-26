"""

Based on https://gist.github.com/felixkreuk/8d70c8c1507fcaac6197d84a8a787fa0
"""

import spur


env = {
    'USE_SIMPLE_THREADED_LEVEL3': '1',
    'OMP_NUM_THREADS': '1',
}
ts = '/PATH-TO-DIR/ts-1.0/ts'


def parallelize(nodes_list, all_runs_args, run_script, on_gpu=False, dry_run=False):
    """
    Running on a list of given servers, a bunch of experiments.
    Assumes that can connect automatically to the servers
    :param nodes_list:
    :param all_runs_args:
    :param run_script:
    :param on_gpu:
    :param dry_run: allows to simply print the intended experiments, and not actually run them
    :return:
    """
    # assumes automatic connection w/o password
    connections = [spur.SshShell(hostname=node, username="USERNAME") for node in nodes_list]

    # ┌──────────────┐
    # │ execute tasks│
    # └──────────────┘

    for sub_exp_idx, combination in enumerate(all_runs_args):
        args_str = f"{ts} sh {run_script}"

        for item in combination:
            args_str += f" {item}"

        if on_gpu:
            gpu_id = sub_exp_idx % 4
            args_str += f" cuda:0"

            node_id = sub_exp_idx // 4 % len(nodes_list)
            env['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"
            env['TS_SOCKET'] = f"/tmp/yanai_gpu_{gpu_id}"
            print(args_str.split(" "), node_id, gpu_id)
        else:
            node_id = sub_exp_idx % len(nodes_list)
            print(args_str.split(" "), node_id)

        if not dry_run:
            connections[node_id].run(args_str.split(" "), update_env=env)

    print(f"==> running {len(all_runs_args)} experiments")
