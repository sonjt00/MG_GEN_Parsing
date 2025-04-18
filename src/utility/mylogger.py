import os
import time

class Logger:
    def __init__(self, result_dir, console_out=True):
        os.makedirs(result_dir, exist_ok=True)
        self.result_dir = result_dir
        self.console_out = console_out
        with open(f"{result_dir}/log.txt", "w") as f:
            f.write("")

    def logging(self, *args, tag="INFO"):
        msg, tag = "", tag.upper().rjust(7) # INFO, ERROR, WARNING
        for arg in args:
            try:
                msg += str(arg) + " "
            except:
                msg += f"[(type:{type(arg)})]"         

        with open(f"{self.result_dir}/log.txt", "a") as f:
            now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log = f"[{now}] {tag}: {msg}"
            f.write(log + '\n')
            if self.console_out:
                print(log)