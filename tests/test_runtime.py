from runtime import RuntimeLog

def func():
    data_path = '../rundata/vpic-runs/skew.16x16.8M.0s0'
    data_path = '../perfstats'
    rt = RuntimeLog(data_path)
    rt.plot_data()

def run():
    func()

if __name__ == '__main__':
    run()