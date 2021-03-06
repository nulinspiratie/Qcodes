"""Qcodes-specific widgets for jupyter notebook."""
import threading
import sys
import os
import time
from IPython.display import display
from ipywidgets.widgets import *
from datetime import datetime
from traitlets import Unicode, Bool

import qcodes as qc
from qcodes.utils.threading import UpdaterThread, raise_exception_in_thread


class LoopManagerWidget(DOMWidget):
    layout = None

    def __init__(self, interval=1):
        super().__init__()

        self.widgets = self.create_widgets()
        self.widget = self.widgets["vbox"]

        self.widgets["pause_button"].on_click(self.pause_measurement)
        self.widgets["stop_button"].on_click(self.stop_measurement)
        self.widgets["force_stop_button"].on_click(self.force_stop_measurement)

        self.widgets["notify_checkbox"].observe(self._handle_notify_button_click)

        self.dot_counter = 0

        if self.layout:
            self.widgets["layout_button"].observe(self._handle_layout_button_click)

        # The layout start/stop button needs to be pressed twice
        # This variable stores the last time it was pressed
        self._layout_button_last_pressed = time.perf_counter()
        # Maximum time difference between successive button presses
        self._layout_button_double_press_dt_max = 0.4

        self.updater = UpdaterThread(self.update_widget, interval=interval)

    def create_widgets(self):
        widgets = {}

        widgets["active_measurement_label"] = Label(value="No active measurement")
        widgets["action_label"] = Label("")
        widgets["loop_indices_label"] = Label("")
        widgets["estimated_completion_time"] = Label("")
        widgets["remaining_time"] = Label("")

        if self.layout:
            widgets["layout_button"] = ToggleButton(
                value=False,
                description="Start Layout",
                disabled=False,
                button_style="success",  # 'success', 'info', 'warning', 'danger' or ''
                tooltip="Description",
                #     icon='check' # (FontAwesome names without the `fa-` prefix)
            )

        widgets["pause_button"] = Button(
            icon="pause", tooltip="Pause measurement", layout=Layout(width="32%")
        )
        widgets["stop_button"] = Button(
            icon="stop", tooltip="Stop measurement", layout=Layout(width="32%")
        )
        widgets["force_stop_button"] = Button(
            icon="stop",
            button_style="danger",
            tooltip="Force stop measurement (not safe)",
            layout=Layout(width="32%"),
        )
        widgets["buttons_hbox"] = HBox(
            [
                widgets["pause_button"],
                widgets["stop_button"],
                widgets["force_stop_button"],
            ]
        )

        widgets["progress_bar"] = FloatProgress(
            value=0,
            min=0,
            max=100,
            step=0.1,
            #  description='Loading:',
            bar_style="info",
            orientation="horizontal",
            layout=Layout(width="96%"),
        )

        # Add notify checkbox
        widgets["notify_checkbox"] = Checkbox(
            False, description="Notify when complete", indent=False
        )

        widgets["vbox"] = VBox(
            [
                widget for name, widget in widgets.items()
                if name not in ["pause_button", "stop_button", "force_stop_button"]
            ]
        )
        return widgets

    def display(self):
        display(self.widget)

    def stop_measurement(self, *args, **kwargs):
        qc.stop()
        self.widgets["stop_button"].disabled = True

    def pause_measurement(self, *args, **kwargs):
        active_measurement = qc.active_measurement()
        if active_measurement.is_paused:
            active_measurement.resume()
            self.widgets["pause_button"].icon = "play"
        else:
            active_measurement.pause()
            self.widgets["pause_button"].icon = "pause"

    def force_stop_measurement(self, *args, **kwargs):
        for thread in threading.enumerate():
            if thread.name == "qcodes_loop":
                raise_exception_in_thread(thread)

    def update_progress_bar(self):
        active_measurement = qc.active_measurement()
        active_dataset = qc.active_dataset()

        if not active_measurement:
            self.widgets["progress_bar"].value = 0
            self.widgets["progress_bar"].description = ""
            return

        if not active_measurement.is_stopped and self.widgets["stop_button"].disabled:
            # Stop button should be enabled
            self.widgets["stop_button"].disabled = False

        if not active_dataset:
            self.widgets["progress_bar"].value = 0
        else:
            # Obtain fraction complete from dataset
            self.widgets["progress_bar"].value = (
                active_dataset.fraction_complete() * 100
            )

            if active_measurement.is_paused:
                # TODO make this working for a Measurement
                if active_measurement.is_paused:
                    self.widgets["progress_bar"].description = (
                        "\u231b " + f"{self.widgets['progress_bar'].value:.0f}%"
                    )
                    self.widgets["progress_bar"].bar_style = "warning"
                else:
                    dots = [
                        "    ",
                        "\u00b7   ",
                        "\u00b7\u00b7  ",
                        "\u00b7\u00b7\u00b7 ",
                    ]
                    self.widgets["progress_bar"].description = (
                        dots[self.dot_counter]
                        + f"{self.widgets['progress_bar'].value:.0f}%"
                    )
                    self.dot_counter = (self.dot_counter + 1) % 4
            else:
                self.widgets[
                    "progress_bar"
                ].description = f"{self.widgets['progress_bar'].value:.0f}%"
                self.widgets["progress_bar"].bar_style = "info"

    def update_widget(self):
        # try:
        # Clear any error messages that mess up the sidebar
        sys.stdout.flush()

        active_measurement = qc.active_measurement()
        active_dataset = qc.active_dataset()

        if not active_measurement or active_dataset is None:
            self.widgets["active_measurement_label"].value = "No active measurement"
            self.widgets["pause_button"].icon = "pause"
            self.widgets["loop_indices_label"].value = ""
            self.widgets["estimated_completion_time"].value = ""
            self.widgets["remaining_time"].value = ""
            self.widgets["action_label"].value = ""
            self.widgets["notify_checkbox"].value = False
        else:
            # Add active dataset name
            dataset_location = f"Active msmt: {active_dataset.location}"
            dataset_name = os.path.split(dataset_location)[-1]
            self.widgets["active_measurement_label"].value = dataset_name

            # Update active action
            action_indices = active_measurement.action_indices
            active_action = active_measurement.active_action_name
            self.widgets[
                "action_label"
            ].value = f"Action {str(action_indices):<9} {active_action}"

            # Update loop indices
            loop_indices = active_measurement.loop_indices
            loop_shape = active_measurement.loop_shape
            self.widgets[
                "loop_indices_label"
            ].value = f"Loop {loop_indices} of {loop_shape}"

            if hasattr(active_dataset, 'metadata') and 't_start' in active_dataset.metadata:
                start_time = datetime.strptime(active_dataset.metadata['t_start'], '%Y-%m-%d %H:%M:%S')
                time_elapsed = datetime.now() - start_time
                remaining_time = time_elapsed * (1/(active_dataset.fraction_complete() + 1e-6) - 1)
                estimated_finish_time = start_time + time_elapsed + remaining_time
                self.widgets[
                    "estimated_completion_time"
                ].value = f"Estimated completion time: {datetime.strftime(estimated_finish_time, '%Y-%m-%d %H:%M')}"
                remaining_hours = remaining_time.total_seconds() // 3600
                remaining_minutes = remaining_time.total_seconds() // 60
                remaining_minutes -= 60 * remaining_hours
                self.widgets["remaining_time"].value = (
                    f"Remaining time: {remaining_hours} hours {remaining_minutes:.1f} minutes"
                )

            # Update notification checkbox
            notify_value = getattr(active_measurement, 'notify', False)
            self.widgets["notify_checkbox"].value = notify_value

        # update layout button
        if self.layout:
            if self.layout.active():
                self.widgets["layout_button"].description = "Stop Layout"
                self.widgets["layout_button"].button_style = "danger"
            else:
                self.widgets["layout_button"].description = "Start Layout"
                self.widgets["layout_button"].button_style = "success"

        self.update_progress_bar()

    def _handle_layout_button_click(self, properties):
        if properties["name"] != "value":
            return

        t_press_diff = time.perf_counter() - self._layout_button_last_pressed
        if t_press_diff > self._layout_button_double_press_dt_max:
            self._layout_button_last_pressed = time.perf_counter()
            self.widgets["layout_button"].value = properties["old"]
        elif properties["new"]:
            self.layout.start()
            self.widgets["layout_button"].description = "Stop Layout"
            self.widgets["layout_button"].button_style = "danger"
        else:
            self.layout.stop()
            self.widgets["layout_button"].description = "Start Layout"
            self.widgets["layout_button"].button_style = "success"

    def _handle_notify_button_click(self, properties):
        if properties["name"] == "value":
            msmt = qc.active_measurement()
            if msmt is not None:
                msmt.notify = properties["new"]
