import glob
import numpy as np
import pandas as pd
import re
import sys
import os


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
    dirname = os.path.basename(run_path)
    params_from_dirname = dirname.split(".")

    all_params = {}
    for p in params_from_dirname:
        match_obj = re.search("(\D+)([0-9]+)", p)
        if not match_obj:
            continue
        param_name = match_obj.group(1)
        param_val = int(match_obj.group(2))
        all_params[param_name] = param_val

    if "nranks" not in all_params:
        nranks = 512
        print(f"Setting default nranks: {nranks} for {run_path}")
        all_params["nranks"] = nranks

    return all_params


def get_run_params(run_path):
    if run_path.endswith("log.txt"):
        run_dir = os.path.dirname(run_path)
        run_log = run_path
    else:
        run_dir = run_path
        run_log = f"{run_dir}/log.txt"

    all_params = get_dirname_params(run_dir)

    with open(run_log, errors="replace") as f:
        lines = f.readlines()
        lines = lines[-200:]

        #  l = [line for line in lines if "per rank" in line][-2]
        #  l = [line for line in lines if "particle writes" in line][0]
        all_l = [
            line
            for line in lines
            if "per rank (min" in line
            and "particle" not in line
            and "dropped" not in line
            and "rpc" not in line
        ]

        for lidx, l in enumerate(all_l):
            mobj = re.search("min: (.*), max: (.*)\)", l)
            wr_min, wr_max = mobj.group(1), mobj.group(2)
            wr_min = humrd_to_num(wr_min)
            wr_max = humrd_to_num(wr_max)
            all_params[f"epoch{lidx + 1}_wr_min"] = wr_min
            all_params[f"epoch{lidx + 1}_wr_max"] = wr_max

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


def gather_alldf_suite_data(all_logs, rundf_abspath):
    all_props = []
    for log in all_logs:
        try:
            log_props = get_run_params(log)
            log_props["runtype"] = os.path.basename(
                os.path.dirname(os.path.dirname(log))
            )
            all_props.append(log_props)
        except Exception as e:
            print(f"Failed to parse: {log}. {str(e)}")

    run_df = pd.DataFrame.from_dict(all_props)
    col_rt = run_df.pop("runtype")
    run_df.insert(0, col_rt.name, col_rt)

    run_df.to_csv(rundf_abspath)
    return run_df


def run_test():
    d = "/mnt/lt20ad1/deltafs-jobdir/network-suite/deltafs.run1.nranks64.epcnt1/log.txt"
    p = get_run_params(d)
    print(p)


def run_allrun_df_suite():
    run_dir = None
    hardcoded_run_dir = "/mnt/lt20ad2/deltafs-jobdir-throttlecheck"
    hardcoded_rundf_dir = "/users/ankushj/repos/carp-root/exp-data/20221004"

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    cwd = os.getcwd()

    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    elif cwd != script_dir:
        run_dir = cwd
    else:
        run_dir = hardcoded_run_dir

    print(f"Using run dir: {run_dir}")

    def glob_level(level):
        glob_str = "/*" * level
        glob_pattern = f"{run_dir}{glob_str}/log.txt"

        print(f"Looking for log.txt files at level {level}... ", end="")
        all_logs = glob.glob(glob_pattern)
        if len(all_logs) == 0:
            print(f"none found")
            return False, all_logs
        else:
            print(f"{len(all_logs)} logs found")
            return True, all_logs

    logs_found, all_logs = glob_level(1)
    if not logs_found:
        logs_found, all_logs = glob_level(2)

    rundf_name = f"{os.path.basename(run_dir)}.csv"
    rundf_abspath = f"{hardcoded_rundf_dir}/{rundf_name}"

    print(f"\nWriting parsed data to {rundf_abspath}")

    run_df = gather_alldf_suite_data(all_logs, rundf_abspath)

    cols_of_int = [
        "runtype",
        "run",
        "nranks",
        "epcnt",
        "intvl",
        "pvtcnt",
        "total_io_time",
    ]

    cols_to_print = [col for col in run_df.columns if col in cols_of_int]
    rundf_print = run_df[cols_to_print].copy()
    rundf_print["total_io_time"] /= 1000.0
    rundf_print = rundf_print.astype({"total_io_time": int})
    rundf_print = rundf_print.sort_values(cols_to_print)
    print(rundf_print)

    return


def run_gen():
    run_allrun_df_suite()


if __name__ == "__main__":
    run_gen()
