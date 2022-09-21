import glob
import numpy as np
import pandas as pd
import re


def humrd_to_num(s):
    s = s.strip()
    s_num = re.findall("[0-9.]+", s)[0]
    num = float(s_num)
    num_x = 1

    if s[-1] == "K":
        num_x = 1000
    elif s[-1] == "M":
        num_x = 1000 * 1000
    else:
        assert s[-1].isdigit()

    num = int(num * num_x)
    return num


def humrd_to_time_ms(s):
    s = s.strip()
    s_unit = re.findall("[a-z]+", s)[0]
    s_num = re.findall("\d+\.\d+", s)[0]
    s_num = float(s_num)

    unitx_map = {"us": 1e-3, "ms": 1, "s": 1e3, "secs": 1e3}

    unitx = unitx_map[s_unit]
    return s_num * unitx


def std(s):
    return np.std(s)


def get_dirname_params(run_path):
    dirname = run_path.split("/")[-1]

    params_from_dirname = dirname.split(".")

    all_params = {}
    for p in params_from_dirname:
        match_obj = re.search("(\D+)([0-9]+)", p)
        param_name = match_obj.group(1)
        param_val = int(match_obj.group(2))
        all_params[param_name] = param_val

    return all_params


def get_run_params(run_path):
    all_params = get_dirname_params(run_path)

    run_log = "{}/log.txt".format(run_path)
    with open(run_log, errors="replace") as f:
        lines = f.readlines()
        lines = lines[-200:]

        #  l = [line for line in lines if "per rank" in line][-2]
        #  l = [line for line in lines if "particle writes" in line][0]
        all_l = [line for line in lines
                 if "per rank (min" in line
                 and "particle" not in line
                 and "dropped" not in line]

        for lidx, l in enumerate(all_l):
            mobj = re.search("min: (.*), max: (.*)\)", l)
            wr_min, wr_max = mobj.group(1), mobj.group(2)
            wr_min = humrd_to_num(wr_min)
            wr_max = humrd_to_num(wr_max)
            all_params[f'epoch{lidx + 1}_wr_min'] = wr_min
            all_params[f'epoch{lidx + 1}_wr_max'] = wr_max

        load_std = 0
        all_l = [line for line in lines if "normalized load stddev" in line]
        if len(all_l):
            mobj = re.search("\d+\.\d+", all_l[0])
            load_std = float(mobj.group(0))

        wr_dropped = 0

        l = [line for line in lines if "particles dropped" in line]
        if len(l) > 0:
            l = l[0]
            mobj = re.search("> (.*) particles", l)
            wr_dropped = humrd_to_num(mobj.group(1))

        all_params["wr_min"] = wr_min
        all_params["wr_max"] = wr_max
        all_params["load_std"] = load_std
        all_params["wr_dropped"] = wr_dropped

        all_l = [line for line in lines if "@ epoch #" in line]
        for l in all_l:
            mobj = re.search("epoch #(\d+)\s+(\d+\.?\d*.*?)- (\d+\.?\d*.*?)\(", l)
            ep_id = mobj.group(1)
            ep_tmin = mobj.group(2)
            ep_tmax = mobj.group(3)
            all_params["epoch{}_tmin".format(ep_id)] = humrd_to_time_ms(ep_tmin)
            all_params["epoch{}_tmax".format(ep_id)] = humrd_to_time_ms(ep_tmax)

    l = [line for line in lines if "final compaction draining" in line][0]
    mobj = re.search("\d+\.\d+.*", l.strip())
    all_params["max_fin_dura"] = humrd_to_time_ms(mobj.group(0))

    l = [line for line in lines if "total io time" in line][0]
    mobj = re.search("\d+\.\d+.*", l.strip())
    all_params["total_io_time"] = humrd_to_time_ms(mobj.group(0))

    return all_params


def gen_allrun_df_deltafs(run_dir):
    run_name = run_dir.split("/")[-1]

    all_runs = glob.glob(run_dir + "/deltafs-baseline*")

    all_params = []
    for run in all_runs:
        try:
            params = get_run_params_deltafs(run)
            all_params.append(params)
        except Exception as e:
            print("Failed to parse: {}".format(run))

    run_df = pd.DataFrame.from_dict(all_params)
    print(run_df)
    run_df.to_csv("{}.csv".format(run_name))


def gen_allrun_df(run_dir, rundf_path):
    all_runs = glob.glob(run_dir + "/run*")

    all_params = []
    for run in all_runs:
        try:
            params = get_run_params(run)
            all_params.append(params)
        except Exception as e:
            print("Failed to parse: {}".format(run))

    run_df = pd.DataFrame.from_dict(all_params)
    print(f"Writing run df to {rundf_path}")
    print(run_df)
    run_df.to_csv(rundf_path)


def run_gen():
    run_dir = "/mnt/lt20ad1/deltafs-jobdir/datascale-runs"

    run_pref = "carp" if "carp-jobdir" in run_dir else "deltafs"
    run_name = run_dir.split("/")[-1]
    rundf_dir = "/users/ankushj/repos/carp-root/exp-data/20220921"
    rundf_path = f"{rundf_dir}/{run_pref}-{run_name}.csv"

    # get_run_params(run_dir)
    gen_allrun_df(run_dir, rundf_path)


if __name__ == "__main__":
    run_gen()
