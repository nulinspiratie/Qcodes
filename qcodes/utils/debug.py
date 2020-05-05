import socket
import time
import re, sys, types, inspect

DEFAULT_LOG = sys.stdout


def formatAllArgs(args, kwds):
    """
    makes a nice string representation of all the arguments
    """
    allargs = []
    for item in args:
        allargs.append(str(item))
    for key, item in kwds.items():
        allargs.append(f"{key}={item}")
    formattedArgs = ", ".join(allargs)
    if len(formattedArgs) > 150:
        return formattedArgs[:146] + " ..."
    return formattedArgs


def logfunction(theFunction, log, displayName=None):
    """Decorator to log whenever a function is called including arguments"""
    if not displayName:
        displayName = theFunction.__name__

    def _wrapper(*args, **kwds):
        argstr = formatAllArgs(args, kwds)

        # Log the entry into the function
        if isinstance(log, socket.socket):
            log.send(f"{displayName}({argstr})\n".encode())
        else:
            print(f"{displayName}({argstr})", file=log)
            log.flush()
            time.sleep(2e-3)

        # t0 = time.perf_counter()
        returnval = theFunction(*args, **kwds)
        # print(f" | {time.perf_counter() - t0:.3g} s", file=log)
        # log.flush()

        # Log return
        ##indentlog("return: %s"% str(returnval)
        return returnval

    return _wrapper


def logmethod(theMethod, log, displayName=None):
    """Decorator to log whenever a method is called including arguments"""

    def _methodWrapper(self, *args, **kwds):
        "Use this one for instance or class methods"

        argstr = formatAllArgs(args, kwds)
        if isinstance(log, socket.socket):
            log.send(f"{displayName}.{theMethod.__name__}({argstr})\n".encode())
        else:
            print(f"{displayName}.{theMethod.__name__}({argstr})", file=log)
            log.flush()
            time.sleep(2e-3)

        # t0 = time.perf_counter()
        returnval = theMethod(self, *args, **kwds)
        # print(f" | {time.perf_counter() - t0:.3g} s", file=log)
        # log.flush()

        return returnval

    return _methodWrapper


def logclass(
    cls, methodsAsFunctions=False, log=None, logMatch=".*", logNotMatch="asdfnomatch"
):
    """
    A class "decorator". But python doesn't support decorator syntax for
    classes, so do it manually::

        class C(object):
           ...
        C = logclass(C)

    @param methodsAsFunctions: set to True if you always want methodname first
    in the display.  Probably breaks if you're using class/staticmethods?
    """
    if log is None:
        log = DEFAULT_LOG

    if not inspect.isclass(cls):
        cls = cls.__class__


    allow = (
        lambda s: re.match(logMatch, s)
        and not re.match(logNotMatch, s)
        and s not in ("__str__", "__repr__")
    )

    namesToCheck = cls.__dict__.keys()

    for name in namesToCheck:
        if not allow(name):
            continue
        # unbound methods show up as mere functions in the values of
        # cls.__dict__,so we have to go through getattr
        value = getattr(cls, name)

        if methodsAsFunctions and callable(value):
            setattr(cls, name, logfunction(value, log=log))
        elif isinstance(value, types.FunctionType) and hasattr(cls, value.__name__):
            setattr(cls, name, logmethod(value, log=log, displayName=cls.__name__))
        elif isinstance(value, types.FunctionType):
            w = logfunction(
                value, log=log, displayName=f"{cls.__name__}.{value.__name__}"
            )
            setattr(cls, name, staticmethod(w))
        elif inspect.ismethod(value) and value.__self__ is cls:
            method = logmethod(value.__func__, log=log, displayName=cls.__name__)
            setattr(cls, name, classmethod(method))

    return cls
