import dataclasses
from datetime import datetime
import inspect
#
#
# @dataclasses.dataclass
# class VerboseFormat:


class verbose:
    # Intentionally leaving verbose lowercase because I think it looks nicer when reading it when implemented.
    def __init__(self, ctx='', show=True, logfile=None, trace=True):

        self.ctx = ctx
        self.show = show
        if logfile is not None:
            self.log = True
            self.logfile = logfile
        else:
            self.log = False
        self.trace = trace
        if trace:
            self._trace_msg()
        else:
            self.strtime = ''
            self.strfile = ''
            self.strfunc = ''
        self.msg = ''

    def __del__(self):
        # And now I am gonna do a sneaky...
        if self.show:
            print(f'{self.strtime}{self.strfile}{self.strfunc}\t{self.msg}\t{self.ctx}')
        if self.log:
            self.logfile.write(f'{self.strtime},{self.strfile},{self.strfunc},{self.msg},{self.ctx}\n')

    def _trace_msg(self, override=''):
        stack = inspect.stack()[2]
        self.strtime = datetime.utcnow().strftime(f'\033[43m{override} UTC %Y-%m-%d %H:%M:%S.%f \033[0m')
        self.strfile = f'\033[44m{override} ' + stack.filename.split('/')[-1] + ' \033[0m'
        self.strfunc = f'\033[45m{override} ' + stack.function + ' \033[0m'

    @staticmethod
    def _underline(word):
        return '\033[4m' + word + '\033[0m'

    @staticmethod
    def _bold(word):
        return '\033[1m' + word + '\033[0m'

    @staticmethod
    def _italic(word):
        return '\033[3m' + word + '\033[0m'

    @staticmethod
    def _strike(word):
        return '\033[9m' + word + '\033[0m'

    @staticmethod
    def _flash(word):
        return '\033[5m' + word + '\033[0m'

    @staticmethod
    def _lesser(word):
        return '\033[2m' + word + '\033[0m'

    @staticmethod
    def _highlight(word):
        return '\033[7m' + word + '\033[0m'

    def warning(self, msg):
        if self.trace:
            self._trace_msg(override='\033[41m')
        self.msg = self._bold('\033[31m' + msg + '\033[0m')

    def trivial(self, msg):
        if self.trace:
            self._trace_msg(override='\033[0m\033[2m')
        self.msg = self._lesser(msg)

    def heading(self, msg):
        if self.trace:
            self._trace_msg(override='\033[42m')
        self.msg = self._bold(self._underline(msg.upper()))

    def message(self, msg):
        self.msg = msg
