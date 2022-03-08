import subprocess

class Command():
    def __init__(self, exec) -> None:
        self.cmd = exec

    def append(self, options):
        if type(options) is type(list()):
            for o in options:
                self.append_with_space(o)
            return
        self.append_with_space(str(options))

    def append_with_space(self, s):
        self.cmd += ' ' + str(s)

    def execute(self):
        try:
            print('execute => \n', self.cmd)
            subprocess.run(
                self.cmd,
                shell=True,
                check=True
                )
        except Exception:
            print("Failed to execute command. Check the output for details")
            raise

    def execute_then_output(self):
        print('execute => \n', self.cmd)
        ret = ''
        try:
            ret = subprocess.check_output(
                self.cmd,
                shell=True,
                stderr=subprocess.STDOUT,
                ).decode("utf-8")
        except Exception:
            print("The command was incorrect. Check the output for details")
            raise
        return ret

    def reset(self):
        self.cmd = self.cmd.split(' ', 1)[0]