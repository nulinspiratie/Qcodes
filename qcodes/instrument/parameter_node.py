import logging
from typing import Sequence, Any, Dict, Callable
import numpy as np
from functools import partial, wraps
from copy import deepcopy

from qcodes.utils.helpers import DelegateAttributes, full_class
from qcodes.utils.metadata import Metadatable
from qcodes.config.config import DotDict
from qcodes.instrument.parameter import _BaseParameter, Parameter
from qcodes.instrument.function import Function

logger = logging.getLogger(__name__)


def parameter(fun):
    def parameter_decorator(self, *args, **kwargs):
        return fun(self, *args, **kwargs)
    return parameter_decorator


parameter_attrs = ['get', 'set', 'vals', 'get_parser', 'set_parser']
class ParameterNodeMetaClass(type):
    def __new__(meta, name, bases, dct):
        dct['_parameter_decorators'] = {}
        # Initialize parameter decorators with those of parent bases
        for base in bases:
            if hasattr(base, '_parameter_decorators'):
                dct['_parameter_decorators'].update(**base._parameter_decorators)

        for attr in list(dct):
            val = dct[attr]
            if getattr(val, '__name__', None) == 'parameter_decorator':
                for parameter_attr in parameter_attrs:
                    if attr.endswith('_'+parameter_attr):
                        parameter_name = attr[:-len(parameter_attr)-1]
                        attr_dict = dct['_parameter_decorators'].get(parameter_name, {})
                        attr_dict[parameter_attr] = val
                        dct['_parameter_decorators'][parameter_name] = attr_dict
                        break
                else:
                    raise SyntaxError('Parameter decorator must end with '
                                      '`_` + {parameter_attr} ')
                dct.pop(attr)
        return super(ParameterNodeMetaClass, meta).__new__(meta, name, bases, dct)


def __deepcopy__(self, memodict={}):
    """Deepcopy method for ParameterNode.

    It is
    """
    # We remove parameters because it may cause circular referencing, i.e. the
    # parameter references the ParameterNode via its decorated method, while
    # the ParameterNode references the parameter via its `parameters` attribute
    restore_attrs = {'__deepcopy__': self.__deepcopy__}
    try:
        for attr in restore_attrs:
            delattr(self, attr)

        self_copy = deepcopy(self)

        for parameter_name, parameter in self_copy.parameters.items():
            if parameter_name in self._parameter_decorators:
                parameter_decorators = self._parameter_decorators[parameter_name]
                self_copy._attach_parameter_decorators(parameter, parameter_decorators)

        return self_copy
    finally:
        for attr_name, attr in restore_attrs.items():
            setattr(self, attr_name, attr)


class ParameterNode(Metadatable, DelegateAttributes, metaclass=ParameterNodeMetaClass):
    """ Container for parameters

    The ParameterNode is a container for `Parameters`, and is primarily used for
    an `Instrument`.

    Args:
        name: Optional name for parameter node

    A parameter can be added to a ParameterNode by settings its attribute:
    ``parameter_node.new_parameter = Parameter()``
    The name of the parameter is set to the attribute name

    Once a parameter has been added its value can be set as such:
    ``parameter_node.new_parameter = 42``
    Note that this doesn't replace the parameter by 42, but instead sets the
    value of the parameter.

     Similarly, its value can be returned by accessing the attribute
     ``parameter_node.new_parameter`` (returns 42)
     Again, this doesn't return the parameter, but its value

    The parameter object can be accessed in two ways:
    - ``parameter_node['new_parameter']``
    - ``parameter_node().new_parameter``
    """

    parameters = {}
    parameter_nodes = {}

    def __init__(self, name: str = None,
                 use_as_attributes: bool = False,
                 **kwargs):
        # Move deepcopy method to the instance scope, since it will temporarily
        # delete its own method during copying (see ParameterNode.__deepcopy__)
        self.__deepcopy__ = partial(__deepcopy__, self)
        self.__copy__ = self.__deepcopy__

        self.use_as_attributes = use_as_attributes
        if name is not None:
            self.name = name

        self.parameters = DotDict()
        self.parameter_nodes = DotDict()
        self.functions = {}
        self.submodules = {}

        super().__init__(**kwargs)

        self._meta_attrs = ['name']

    def __repr__(self):
        repr_str = 'ParameterNode '
        if hasattr(self, 'name'):
            if isinstance(self.name, _BaseParameter):
                repr_str += f'{self.name()} '
            else:
                repr_str += f'{self.name} '
        repr_str += 'containing '
        if self.parameter_nodes:
            repr_str += f'{len(parameter_nodes)} nodes, '
        repr_str += f'{len(self.parameters)} parameters'
        return repr_str

    def __getattr__(self, attr):
        if attr == 'use_as_attributes':
            return super().__getattr__(attr)
        elif attr in self.parameter_nodes:
            return self.parameter_nodes[attr]
        elif attr in self.parameters:
            parameter = self.parameters[attr]
            if self.use_as_attributes:
                # Perform get and return value
                return parameter()
            else:
                # Return parameter instance
                return parameter
        else:
            return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if isinstance(val, _BaseParameter):
            self.parameters[attr] = val

            if attr in self._parameter_decorators:
                # Some methods have been defined in the ParameterNode as methods
                # Using the @parameter decorator.
                self._attach_parameter_decorators(
                    parameter=val,
                    decorator_methods=self._parameter_decorators[attr])
            if val.name == 'None':
                # Parameter has been created without name, update name to attr
                val.name = attr
                if val.label is None:
                    # For label, convert underscores to spaces and capitalize
                    label = attr.replace('_', ' ')
                    label = label[0].capitalize() + label[1:]
                    val.label = label
        elif isinstance(val, ParameterNode):
            self.parameter_nodes[attr] = val
            if not hasattr(val, 'name'):
                # Update nested ParameterNode name
                val.name = attr
        elif attr in self.parameters:
            # Set parameter value
            self.parameters[attr](val)
        else:
            super().__setattr__(attr, val)

    def _attach_parameter_decorators(self,
                                     parameter: _BaseParameter,
                                     decorator_methods: Dict[str, Callable]):
        """Attaches @parameter decorators to a parameter

        Args:
            parameter: Parameter to attach decorators to
            decorator_methods: Decorator methods to attach to parameter
            """
        for param_attr, param_method in decorator_methods.items():
            method_with_args = partial(param_method, self, parameter)
            if param_attr == 'get':
                parameter.get_raw = method_with_args
                if parameter.wrap_get:
                    parameter.get = parameter._wrap_get(parameter.get_raw)
                else:
                    parameter.get = parameter.get_raw
            elif param_attr == 'set':
                parameter.set_raw = method_with_args
                if parameter.wrap_set:
                    parameter.set = parameter._wrap_set(parameter.set_raw)
                else:
                    parameter.set = parameter.set_raw
            else:
                setattr(parameter, param_attr, method_with_args)
        # perform a set without eparameteruating, which saves the value,
        # ensuring that new modifications such as the set_parser are
        # taken into account
        if hasattr(parameter, 'set') and parameter.raw_value is not None:
            parameter.set(parameter.get_latest(), evaluate=False)

    __copy__ = deepcopy

    # def __dir__(self):
    #     # Add parameters to dir
    #     items = super().__dir__()
    #     items.extend(self.parameters.keys())
    #     return items

    def add_function(self, name: str, **kwargs):
        """ Bind one Function to this parameter node.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real function of the instrument.

        This functionality is meant for simple cases, principally things that
        map to simple commands like '\*RST' (reset) or those with just a few
        arguments. It requires a fixed argument count, and positional args
        only. If your case is more complicated, you're probably better off
        simply making a new method in your ``Instrument`` subclass definition.

        Args:
            name: how the Function will be stored within
            ``instrument.Functions`` and also how you  address it using the
            shortcut methods: ``parameter_node.call(func_name, *args)`` etc.

            **kwargs: constructor kwargs for ``Function``

        Raises:
            KeyError: if this instrument already has a function with this
                name.
        """
        if name in self.functions:
            raise KeyError('Duplicate function name {}'.format(name))
        func = Function(name=name, instrument=self, **kwargs)
        self.functions[name] = func

    def add_submodule(self, name: str, submodule: Metadatable):
        """ Bind one submodule to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        method for every submodule of the instrument.

        Submodules can effectively be considered as instruments within the main
        instrument, and should at minimum be snapshottable. For example, they
        can be used to either store logical groupings of parameters, which may
        or may not be repeated, or channel lists.

        Args:
            name: how the submodule will be stored within
                `instrument.submodules` and also how it can be addressed.

            submodule: The submodule to be stored.

        Raises:
            KeyError: if this instrument already contains a submodule with this
                name.
            TypeError: if the submodule that we are trying to add is not an
                instance of an Metadatable object.
        """
        if name in self.submodules:
            raise KeyError('Duplicate submodule name {}'.format(name))
        if not isinstance(submodule, Metadatable):
            raise TypeError('Submodules must be metadatable.')
        self.submodules[name] = submodule

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str]=None):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)

        Returns:
            dict: base snapshot
        """

        snap = {
            "functions": {name: func.snapshot(update=update)
                          for name, func in self.functions.items()},
            "submodules": {name: subm.snapshot(update=update)
                           for name, subm in self.submodules.items()},
            "__class__": full_class(self)
        }

        snap['parameters'] = {}
        for name, param in self.parameters.items():
            update = update
            if params_to_skip_update and name in params_to_skip_update:
                update = False
            try:
                snap['parameters'][name] = param.snapshot(update=update)
            except:
                logging.info("Snapshot: Could not update parameter:", name)
                snap['parameters'][name] = param.snapshot(update=False)
        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        return snap

    def print_snapshot(self,
                       update: bool = False,
                       max_chars: int = 80):
        """ Prints a readable version of the snapshot.

        The readable snapshot includes the name, value and unit of each
        parameter.
        A convenience function to quickly get an overview of the parameter node.

        Args:
            update: If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
                This argument gets passed to the snapshot function.
            max_chars: the maximum number of characters per line. The
                readable snapshot will be cropped if this value is exceeded.
                Defaults to 80 to be consistent with default terminal width.
        """
        floating_types = (float, np.integer, np.floating)
        snapshot = self.snapshot(update=update)

        par_lengths = [len(p) for p in snapshot['parameters']]

        # Min of 50 is to prevent a super long parameter name to break this
        # function
        par_field_len = min(max(par_lengths)+1, 50)

        if hasattr(self, 'name'):
            print(str(self.name)+ ':')
        print('{0:<{1}}'.format('\tparameter ', par_field_len) + 'value')
        print('-'*max_chars)
        for par in sorted(snapshot['parameters']):
            name = snapshot['parameters'][par]['name']
            msg = '{0:<{1}}:'.format(name, par_field_len)

            # in case of e.g. ArrayParameters, that usually have
            # snapshot_value == False, the parameter may not have
            # a value in the snapshot
            val = snapshot['parameters'][par].get('value', 'Not available')

            unit = snapshot['parameters'][par].get('unit', None)
            if unit is None:
                # this may be a multi parameter
                unit = snapshot['parameters'][par].get('units', None)
            if isinstance(val, floating_types):
                msg += '\t{:.5g} '.format(val)
            else:
                msg += '\t{} '.format(val)
            if unit is not '':  # corresponds to no unit
                msg += '({})'.format(unit)
            # Truncate the message if it is longer than max length
            if len(msg) > max_chars and not max_chars == -1:
                msg = msg[0:max_chars-3] + '...'
            print(msg)

        for submodule in self.submodules.values():
            if hasattr(submodule, '_channels'):
                if submodule._snapshotable:
                    for channel in submodule._channels:
                        channel.print_readable_snapshot()
            else:
                submodule.print_readable_snapshot(update, max_chars)

    #
    # shortcuts to parameters & setters & getters                           #
    # instrument.someparam === instrument.parameters['someparam']           #
    # instrument.get('someparam') === instrument['someparam'].get()         #
    # etc...                                                                #
    #
    delegate_attr_dicts = ['parameters', 'parameter_nodes', 'functions',
                           'submodules']

    def __getitem__(self, key):
        """Delegate instrument['name'] to parameter or function 'name'."""
        try:
            return self.parameters[key]
        except KeyError:
            pass
        try:
            return self.functions[key]
        except KeyError:
            pass
        return super().__getitem__(key)

    def call(self, func_name: str, *args, **kwargs):
        """ Shortcut for calling a function from its name.

        Args:
            func_name: The name of a function of this instrument.
            *args: any arguments to the function.
            **kwargs: any keyword arguments to the function.

        Returns:
            any: The return value of the function.
        """
        return self.functions[func_name].call(*args, **kwargs)

    def validate_status(self,
                        verbose: bool = False):
        """ Validate the values of all gettable parameters

        The validation is done for all parameters that have both a get and
        set method.

        Arguments:
            verbose: If True, information of checked parameters is printed.

        """
        for k, p in self.parameters.items():
            if hasattr(p, 'get') and hasattr(p, 'set'):
                value = p.get()
                if verbose:
                    print('validate_status: param %s: %s' % (k, value))
                p.validate(value)

    def print_readable_snapshot(self, update=False, max_chars=80):
        logger.warning('print_readable_snapshot is replaced with print_snapshot')
        self.print_snapshot(update=update, max_chars=max_chars)

    def add_parameter(self, name, parameter_class=Parameter, **kwargs):
        """
        Bind one Parameter to this instrument.

        Instrument subclasses can call this repeatedly in their ``__init__``
        for every real parameter of the instrument.

        In this sense, parameters are the state variables of the instrument,
        anything the user can set and/or get

        Args:
            name (str): How the parameter will be stored within
                ``instrument.parameters`` and also how you address it using the
                shortcut methods: ``instrument.set(param_name, value)`` etc.

            parameter_class (Optional[type]): You can construct the parameter
                out of any class. Default ``StandardParameter``.

            **kwargs: constructor arguments for ``parameter_class``.

        Raises:
            KeyError: if this instrument already has a parameter with this
                name.
        """
        if name in self.parameters:
            raise KeyError('Duplicate parameter name {}'.format(name))
        param = parameter_class(name=name, instrument=self, **kwargs)
        self.parameters[name] = param