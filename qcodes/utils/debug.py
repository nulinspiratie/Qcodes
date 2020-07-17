"""Debugging tools

The main usage in this module is:
- Modify a class such that every method call sends out a log message
"""
from typing import Callable
import socket
import time
import re, sys, types, inspect

DEFAULT_LOG = sys.stdout


def format_all_args(args, kwds):
    """makes a nice string representation of all the arguments"""
    allargs = []
    for item in args:
        allargs.append(str(item))
    for key, item in kwds.items():
        allargs.append(f"{key}={item}")
    formattedArgs = ", ".join(allargs)
    if len(formattedArgs) > 150:
        return formattedArgs[:146] + " ..."
    return formattedArgs


def log_function(
        logged_function: Callable,
        log=None,
        display_name=None
):
    """Decorator to log whenever a function is called including arguments

    Args:
        logged_function: Function that when called emits a log message
        log: a file-like object (stream), such as sys.stdout.
            Can also be a socket.
            Default is qcodes.utils.debug.DEFAULT_LOG.
        display_name: Optional name. Default is function name

    Example:
        def fun(x):
            return x + 1

        fun_logged = log_function(fun)
        fun_logged(2);
        # Prints "fun(2)"
    """
    if log is None:
        log = DEFAULT_LOG

    if not display_name:
        display_name = logged_function.__name__

    def _wrapper(*args, **kwds):
        argstr = format_all_args(args, kwds)

        # Log the entry into the function
        if isinstance(log, socket.socket):
            log.send(f"{display_name}({argstr})\n".encode())
        else:
            print(f"{display_name}({argstr})", file=log)
            log.flush()

        returnval = logged_function(*args, **kwds)
        return returnval

    return _wrapper


def log_method(log_method, log=None, display_name=''):
    """Decorator to log whenever a method is called including arguments

    Main usage is in function log_class

    Args:
        log_method: Class method that when called emits a log message
        log: a file-like object (stream), such as sys.stdout.
            Can also be a socket.
            Default is qcodes.utils.debug.DEFAULT_LOG.
        display_name: Optional name of class. Default is None

    Example:
        class C:
            def fun(self, x):
                return x+1

        c = C()
        method_logged = log_method(C.fun, display_name="c")
        method_logged(c, 123)
        # Prints "c.fun(123)"
    """
    if log is None:
        log = DEFAULT_LOG

    if display_name:
        display_name += '.'

    def _methodWrapper(self, *args, **kwds):
        "Use this one for instance or class methods"

        argstr = format_all_args(args, kwds)
        if isinstance(log, socket.socket):
            log.send(f"{display_name}{log_method.__name__}({argstr})\n".encode())
        else:
            print(f"{display_name}{log_method.__name__}({argstr})", file=log)
            log.flush()

        returnval = log_method(self, *args, **kwds)

        return returnval

    return _methodWrapper


def log_class(
        cls,
        methods_as_functions=False,
        log=None,
        log_match=".*",
        log_not_match="asdfnomatch"
):
    """Class decorator to log every method call of a class.

    Since python doesn't support decorator syntax, it has to be applied manually:

        class C(object):
           ...
        C = logclass(C)

    Args:
        cls: class / object to decorate such that all method calls emit log.
        methods_as_functions: set to True if you always want methodn_ame first
            in the display.  Probably breaks if you're using class/staticmethods?
        log: a file-like object (stream), such as sys.stdout.
            Can also be a socket.
            Default is qcodes.utils.debug.DEFAULT_LOG.
        log_match: RegEx string representation that each method of the class
            must satisfy to be logged
        log_not_match: Optional RegEx string representation, for which if the
            method each method of the class
            must satisfy to be logged

    """
    if log is None:
        log = DEFAULT_LOG

    if not inspect.isclass(cls):
        cls = cls.__class__

    allow = (
        lambda s: re.match(log_match, s)
        and (log_not_match is None or not re.match(log_not_match, s))
        and s not in ("__str__", "__repr__")
    )

    names_to_check = cls.__dict__.keys()

    for name in names_to_check:
        if not allow(name):
            continue
        # unbound methods show up as mere functions in the values of
        # cls.__dict__,so we have to go through getattr
        value = getattr(cls, name)

        if methods_as_functions and callable(value):
            setattr(cls, name, log_function(value, log=log))
        elif isinstance(value, types.FunctionType) and hasattr(cls, value.__name__):
            setattr(cls, name, log_method(value, log=log, display_name=cls.__name__))
        elif isinstance(value, types.FunctionType):
            w = log_function(
                value, log=log, display_name=f"{cls.__name__}.{value.__name__}"
            )
            setattr(cls, name, staticmethod(w))
        elif inspect.ismethod(value) and value.__self__ is cls:
            method = log_method(value.__func__, log=log, display_name=cls.__name__)
            setattr(cls, name, classmethod(method))

    return cls
