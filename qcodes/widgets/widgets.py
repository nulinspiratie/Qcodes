"""Qcodes-specific widgets for jupyter notebook."""
import threading
import sys
import os
from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

import qcodes as qc
from qcodes.utils.threading import UpdaterThread, raise_exception_in_thread


class LoopManagerWidget(DOMWidget):
    def __init__(self, interval=1):
        super().__init__()

        self.create_widgets()

        self.pause_button.on_click(self.pause_measurement)
        self.stop_button.on_click(self.stop_measurement)
        self.force_stop_button.on_click(self.force_stop_measurement)

        self.dot_counter = 0

        self.updater = UpdaterThread(self.update_widget, interval=interval)

    def create_widgets(self):

        self.active_measurement_label = Label(value="No active measurement")
        self.action_label = Label("")
        self.loop_indices_label = Label("")

        self.pause_button = Button(
            icon="pause", tooltip="Pause measurement", layout=Layout(width="32%")
        )
        self.stop_button = Button(
            icon="stop", tooltip="Stop measurement", layout=Layout(width="32%")
        )
        self.force_stop_button = Button(
            icon="stop",
            button_style="danger",
            tooltip="Force stop measurement (not safe)",
            layout=Layout(width="32%"),
        )
        self.buttons_hbox = HBox(
            [self.pause_button, self.stop_button, self.force_stop_button]
        )

        self.progress_bar = FloatProgress(
            value=0,
            min=0,
            max=100,
            step=0.1,
            #             description='Loading:',
            bar_style="info",
            orientation="horizontal",
            layout=Layout(width="96%"),
        )

        self.vbox = VBox(
            [
                self.active_measurement_label,
                self.action_label,
                self.loop_indices_label,
                self.buttons_hbox,
                self.progress_bar,
            ]
        )

    def display(self):
        display(self.vbox)

    def stop_measurement(self, *args, **kwargs):
        qc.stop()
        self.stop_button.disabled = True

    def pause_measurement(self, *args, **kwargs):
        if qc.active_measurement().is_paused:
            qc.active_measurement().resume()
            self.pause_button.icon = "play"
        else:
            qc.active_measurement().pause()
            self.pause_button.icon = "pause"

    def force_stop_measurement(self, *args, **kwargs):
        for thread in threading.enumerate():
            if thread.name == "qcodes_loop":
                raise_exception_in_thread(thread)

    def update_progress_bar(self):
        active_measurement = qc.active_measurement()
        active_dataset = qc.active_dataset()

        if not active_measurement:
            self.progress_bar.value = 0
            self.progress_bar.description = ""
            return

        if not active_measurement.is_stopped and self.stop_button.disabled:
            # Stop button should be enabled
            self.stop_button.disabled = False

        if not active_dataset:
            self.progress_bar.value = 0
        else:
            # Obtain fraction complete from dataset
            self.progress_bar.value = active_dataset.fraction_complete() * 100

            if active_measurement.is_paused:
                # TODO make this working for a Measurement
                if active_measurement.is_paused:
                    self.progress_bar.description = (
                        "\u231b " + f"{self.progress_bar.value:.0f}%"
                    )
                    self.progress_bar.bar_style = "warning"
                else:
                    dots = [
                        "    ",
                        "\u00b7   ",
                        "\u00b7\u00b7  ",
                        "\u00b7\u00b7\u00b7 ",
                    ]
                    self.progress_bar.description = (
                        dots[self.dot_counter] + f"{self.progress_bar.value:.0f}%"
                    )
                    self.dot_counter = (self.dot_counter + 1) % 4
            else:
                self.progress_bar.description = f"{self.progress_bar.value:.0f}%"
                self.progress_bar.bar_style = "info"

    def update_widget(self):
        # try:
            # Clear any error messages that mess up the sidebar
        sys.stdout.flush()

        if not qc.active_measurement():
            self.active_measurement_label.value = "No active measurement"
            self.pause_button.icon = "pause"
            self.loop_indices_label.value = ""
            self.action_label.value = ""
        else:
            # Add active dataset name
            dataset_location = f"Active msmt: {qc.active_dataset().location}"
            dataset_name = os.path.split(dataset_location)[-1]
            self.active_measurement_label.value = dataset_name

            # Update active action
            action_indices = qc.active_measurement().action_indices
            active_action = qc.active_measurement().active_action_name
            self.action_label.value = f"Action {str(action_indices):<9} {active_action}"

            # Update loop indices
            loop_indices = qc.active_measurement().loop_indices
            loop_shape = qc.active_measurement().loop_shape
            self.loop_indices_label.value = f"Loop {loop_indices} of {loop_shape}"

        self.update_progress_bar()

    # except:
    #     raise
