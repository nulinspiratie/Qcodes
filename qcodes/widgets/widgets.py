"""Qcodes-specific widgets for jupyter notebook."""
import threading
import os
from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

from .display import display_auto

display_auto('widgets/sidebar_widget.js')


class SidebarWidget(DOMWidget):
    _view_name = Unicode('SidebarView').tag(sync=True)
    _view_module = Unicode('sidebar').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _widget_name = Unicode('none').tag(sync=True)
    _add_widget = Bool(False).tag(sync=True)
    _playing = Bool(False).tag(sync=True)

    def __init__(self):
        super().__init__()
        display(self)

    def add_widget(self, widget_name):
        self._widget_name = widget_name
        self._add_widget = not self._add_widget


import qcodes as qc
from qcodes.utils.threading import UpdaterThread


class LoopManagerWidget(DOMWidget):
    def __init__(self, interval=1):
        super().__init__()
        self.active_loop_label = Label(value='No active loop')

        self.stop_button = Button(icon='stop', tooltip='Stop measurement')
        self.force_stop_button = Button(icon='stop', button_style='danger',
                                        tooltip='Force stop measurement (not safe)')
        self.buttons_hbox = HBox([self.stop_button, self.force_stop_button])

        self.progress_bar = FloatProgress(
            value=0,
            min=0,
            max=100,
            step=0.1,
            #             description='Loading:',
            bar_style='info',
            orientation='horizontal',
            layout=Layout(width='95%')
        )

        self.vbox = VBox([self.active_loop_label,
                          self.buttons_hbox,
                          self.progress_bar])

        self.stop_button.on_click(self.stop_loop)
        self.force_stop_button.on_click(self.force_stop_loop)

        self.updater = UpdaterThread(self.update_widget,
                                     interval=interval)

    def display(self):
        display(self.vbox)

    def stop_loop(self, *args, **kwargs):
        qc.stop()
        self.stop_button.disabled = True

    def force_stop_loop(self, *args, **kwargs):
        for thread in threading.enumerate():
            if thread.name == 'qcodes_loop':
                thread.terminate()

    def update_widget(self):
        try:
            import sys
            sys.stdout.flush()
            if not qc.active_loop():
                self.active_loop_label.value = 'No active loop'
            else:
                dataset_location = f'Active loop: {qc.active_data_set().location}'
                dataset_name = os.path.split(dataset_location)[-1]
                self.active_loop_label.value = dataset_name

            if not qc.loops.ActiveLoop._is_stopped and self.stop_button.disabled:
                self.stop_button.disabled = False
            if not qc.active_data_set():
                self.progress_bar.value = 0
            else:
                self.progress_bar.value = qc.active_data_set().fraction_complete() * 100
        except:
            pass