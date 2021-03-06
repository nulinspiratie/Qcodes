"""Actions, mainly to be executed in measurement Loops."""
import time
import logging

from qcodes.utils.deferred_operations import is_function
from qcodes.utils.threading import thread_map


logger = logging.getLogger(__name__)

_NO_SNAPSHOT = {'type': None, 'description': 'Action without snapshot'}


# exception when threading is attempted used to simultaneously
# query the same instrument for several values
class UnsafeThreadingException(Exception):
    pass


def _actions_snapshot(actions, update):
    """Make a list of snapshots from a list of actions."""
    snapshot = []
    for action in actions:
        if hasattr(action, 'snapshot'):
            snapshot.append(action.snapshot(update=update))
        else:
            snapshot.append(_NO_SNAPSHOT)
    return snapshot


class Task:
    """
    A predefined task to be executed within a measurement Loop.

    The first argument should be a callable, to which any subsequent
    args and kwargs (which are evaluated before the loop starts) are passed.

    The args and kwargs are first evaluated if they are found to be callable.

    Keyword Args passed when the Task is called are ignored,
    but are accepted for compatibility with other things happening in a Loop.

    Args:
        func (callable): Function to executed
        N (int): if set, Task is only executed every N calls.
            Task is always executed in first call.
        condition (callable): If set, Task is only executed if condition
            evaluates to True
        *args: pass to func, after evaluation if callable
        **kwargs: pass to func, after evaluation if callable

    """

    def __init__(self, func, *args, N=None, condition=None, repetitions=1, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        self.N = N
        self.idx = 0
        self.repetitions = repetitions

        self.condition = condition

    def __call__(self, **ignore_kwargs):
        if self.N is not None:
            if self.idx % self.N != 0:
                self.idx += 1
                return
            self.idx += 1

        if self.condition is not None:
            if not self.condition():
                return

        # If any of the arguments are callable, evaluate them first
        eval_args = [arg() if callable(arg) else arg for arg in self.args]
        eval_kwargs = {k: (v() if callable(v) else v) for k, v in
                       self.kwargs.items()}

        for repetition in range(self.repetitions):
            self.func(*eval_args, **eval_kwargs)

    def snapshot(self, update=False):
        """
        Snapshots  task
        Args:
            update (bool): TODO not in use

        Returns:
            dict: snapshot
        """
        snapshot = {'type': 'Task', 'func': repr(self.func)}
        if self.N is not None:
            snapshot['N'] = self.N
            snapshot['idx'] = self.idx
        if self.condition is not None:
            snapshot['condition'] = self.condition
        return snapshot


class Wait:
    """
    A simple class to tell a Loop to wait <delay> seconds.

    This is transformed into a Task within the Loop, such that
    it can do other things (monitor, check for halt) during the delay.

    But for use outside of a Loop, it is also callable (then it just sleeps)

    Args:
        delay: seconds to delay

    Raises:
        ValueError: if delay is negative
    """
    def __init__(self, delay):
        if not delay >= 0:
            raise ValueError('delay must be > 0, not {}'.format(repr(delay)))
        self.delay = delay

    def __call__(self):
        if self.delay:
            time.sleep(self.delay)

    def snapshot(self, update=False):
        """
        Snapshots  delay
        Args:
            update (bool): TODO not in use

        Returns:
            dict: snapshot
        """
        return {'type': 'Wait', 'delay': self.delay}


class _Measure:
    """
    A callable collection of parameters to measure.

    This should not be constructed manually, only by an ActiveLoop.
    """
    def __init__(self, params_indices, data_set, use_threads):
        self.use_threads = use_threads and len(params_indices) > 1
        # the applicable DataSet.store function
        self.store = data_set.store

        # for performance, pre-calculate which params return data for
        # multiple arrays, and the name mappings
        self.parameters = []
        self.param_ids = []
        self.composite = []
        paramcheck = []  # list to check if parameters are unique
        for param, action_indices in params_indices:
            self.parameters.append(param)

            if param._instrument:
                paramcheck.append((param, param._instrument))

            if hasattr(param, 'names'):
                part_ids = []
                for i in range(len(param.names)):
                    param_id = data_set.action_id_map[action_indices + (i,)]
                    part_ids.append(param_id)
                self.param_ids.append(None)
                self.composite.append(part_ids)
            else:
                param_id = data_set.action_id_map[action_indices]
                self.param_ids.append(param_id)
                self.composite.append(False)

        if self.use_threads:
            insts = [p[1] for p in paramcheck]
            if (len(set(insts)) != len(insts)):
                duplicates = [p for p in paramcheck if insts.count(p[1]) > 1]
                raise UnsafeThreadingException('Can not use threading to '
                                               'read '
                                               'several things from the same '
                                               'instrument. Specifically, you '
                                               'asked for'
                                               ' {}.'.format(duplicates))

    def __call__(self, action_indices, current_action_idx, loop_indices, **ignore_kwargs):
        from qcodes.loops import ActiveLoop

        out_dict = {}
        if self.use_threads:
            out = thread_map(self.parameters)
        else:
            out = []
            for k, parameter in enumerate(self.parameters):
                # Registering action as active one in case it needs to be
                # accessed during the action itself
                ActiveLoop.action_indices = action_indices + (current_action_idx + k,)
                ActiveLoop.active_action = parameter
                out.append(parameter.get())

        for param_out, param_id, composite in zip(out, self.param_ids,
                                                  self.composite):
            if composite:
                for val, part_id in zip(param_out, composite):
                    out_dict[part_id] = val
            else:
                out_dict[param_id] = param_out
        self.store(loop_indices, out_dict)


class _Nest:

    """
    Wrapper to make a callable nested ActiveLoop.

    This should not be constructed manually, only by an ActiveLoop.
    """

    def __init__(self, inner_loop, action_indices):
        self.inner_loop = inner_loop
        self.action_indices = action_indices

    def __call__(self, action_indices, **kwargs):
        # We ignore the passed action indices
        self.inner_loop._run_loop(action_indices=self.action_indices, **kwargs)


class BreakIf:

    """
    Loop action that breaks out of the loop if a condition is truthy.

    Args:
        condition (callable): a callable taking no arguments.
            Can be a simple function that returns truthy when it's time to quit
            May also be constructed by deferred operations on `Parameter`.
    Raises:
        TypeError: if condition is not a callable with no aguments.

    Examples:
            >>> BreakIf(gates.chan1 >= 3)
            >>> BreakIf(abs(source.I * source.V) >= source.power_limit.get_latest)
    """

    def __init__(self, condition):
        if not is_function(condition, 0):
            raise TypeError('BreakIf condition must be a callable with '
                            'no arguments')
        self.condition = condition

    def __call__(self, **ignore_kwargs):
        if self.condition():
            logger.info(f'Breakif Condition not met: {self.condition}')
            raise _QcodesBreak

    def snapshot(self, update=False):
        # TODO: make nice reprs for DeferredOperations
        """
        Snapshots breakif action
        Args:
            update (bool): TODO not in use

        Returns:
            dict: snapshot

        """
        return {'type': 'BreakIf', 'condition': repr(self.condition)}


class SkipIf(BreakIf):
    """Perform continue, i.e. proceed to next loop, if condition is met.

    Behaviour is similar to BreakIf
    """
    pass


class ContinueIf(SkipIf):
    """Perform continue, i.e. proceed to next loop, if condition is met.

    Behaviour is similar to BreakIf
    """
    def __call__(self, **ignore_kwargs):
        logger.warning('Should use SkipIf instead')
        super().__call__(**ignore_kwargs)


class _QcodesBreak(Exception):
    pass
