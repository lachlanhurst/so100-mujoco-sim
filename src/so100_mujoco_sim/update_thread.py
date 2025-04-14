import time

import mujoco
from PySide6.QtCore import QThread, Signal

from so100_mujoco_sim.arm_control import (
    ArmController,
    MujocoArmController,
    So100ArmController,
    PlaybackRecordController,
    PlaybackRecordState,
    UiArmController,
    update_from_controller
)


class UpdateThread(QThread):

    update_ui_joint_values = Signal(list)
    update_ui_recorded_steps = Signal(int, int)
    update_controller_enabled_states = Signal()
    update_primary_controller = Signal(str)
    warning = Signal(str, str)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.mujoco_controller = MujocoArmController(model, data)
        # user interface controller (the UI sliders)
        # reuse the mujoco model joint definition
        self.ui_controller = UiArmController(self.mujoco_controller.joints, self._update_ui)
        self.ui_controller.primary = True
        self._primary_controller_index = 0
        self.real_controller = So100ArmController()
        self.playback_record_controller = PlaybackRecordController(
            self.mujoco_controller.joints,
            self._update_ui_recorded_steps
        )

        self.arm_controllers: list[ArmController] = []
        self.arm_controllers.append(self.ui_controller)
        self.arm_controllers.append(self.mujoco_controller)
        self.arm_controllers.append(self.real_controller)
        self.arm_controllers.append(self.playback_record_controller)

        # reset the simulation timer
        self.reset()

        self._playback_file = None
        self._playback_file_save = False
        self._playback_file_load = False

        self._do_connection = False
        self.running = True

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def get_primary_controller(self) -> ArmController | None:
        for ac in self.arm_controllers:
            if ac.primary:
                return ac
        return None

    def run(self) -> None:
        while self.running:
            # don't step the simulation past real time
            # without this the sim usually finishes before it's
            # even visible
            if self.data.time < self.real_time / 1_000_000_000:
                # Update the control loop at a 100hz
                if (time.monotonic_ns() - self.last_robot_update) / 1_000_000_000 >= (1/100):
                    self.last_robot_update = time.monotonic_ns()

                    if self.get_primary_controller_index() != self._primary_controller_index:
                        self._set_primary_controller_index(self._primary_controller_index)
                        self.update_primary_controller.emit(
                            self.arm_controllers[self._primary_controller_index].name
                        )

                    pc = self.get_primary_controller()
                    for ac in self.arm_controllers:
                        # update the actual positions of the controller
                        ac.update()
                        if ac.primary:
                            pass
                        else:
                            # set each of the secondary controllers set positions to that of
                            # the primary's output
                            update_from_controller(pc, ac)
                            # now send those positions to the controller
                            ac.set_positions()

                # step the simulation
                mujoco.mj_step(self.model, self.data)

                # here's where we check if there's something that was requested
                # to be done from the UI thread, and do it
                if self._do_connection:
                    self._connect_real_robot()
                if self._playback_file_save:
                    self._save_playback_file()
                if self._playback_file_load:
                    self._load_playback_file()
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.mujoco_controller.reset()

    def set_joint_position(self, joint_name: str, position: float) -> None:
        self.ui_controller.set_joint_set_position(joint_name, position)

    def get_controller_names(self) -> str:
        return [c.name for c in self.arm_controllers]

    def get_controllable_controllers(self) -> list[bool]:
        return [c.controllable for c in self.arm_controllers]

    def get_primary_controller_index(self) -> int:
        for i, c in enumerate(self.arm_controllers):
            if c.primary:
                return i
        return 0

    def set_primary_controller_index(self, index: int) -> int:
        # we need to make sure the primary flag on the controllers
        # is changed in the main update loop as  setting this flag
        # can send commands to the motors (which may be getting sent
        # data from the update thread). So set the desired index
        # here, and do the proper update for thread in the following fn
        self._primary_controller_index = index

    def _set_primary_controller_index(self, index: int) -> int:
        for i, c in enumerate(self.arm_controllers):
            if i == index:
                c.primary = True
            else:
                c.primary = False

    def connect_real_robot(self, usb_port: str, calibration_folder: str) -> None:
        self._usb_port = usb_port
        self._calibration_folder = calibration_folder
        self._do_connection = True

    def _connect_real_robot(self) -> None:
        if not self._do_connection:
            return
        self._do_connection = False

        try:
            self.real_controller.connect(self._usb_port, self._calibration_folder)
            self.real_controller.update()
        except Exception as e:
            self.warning.emit("Connection issue", f"Error connecting to real robot: {e}")
            return

        # this updates the ui, but also raises change events that causes the mujoco
        # model to update
        self.update_ui_joint_values.emit(self.real_controller.joint_output_positions)
        # raise event to tell UI that the real robot controller can be enabled
        self.update_controller_enabled_states.emit()

    def _update_ui(self) -> None:
        self.update_ui_joint_values.emit(self.ui_controller.joint_set_positions)

    def set_playback_record_state(self, state: PlaybackRecordState) -> None:
        self.playback_record_controller.set_state(state)

    def _update_ui_recorded_steps(self) -> None:
        self.update_ui_recorded_steps.emit(
            len(self.playback_record_controller.recorded_joint_positions),
            self.playback_record_controller.playback_index
        )

    def save_playback_file(self, file_name: str) -> None:
        self._playback_file = file_name
        self._playback_file_save = True

    def _save_playback_file(self) -> None:
        if self._playback_file_save:
            self.playback_record_controller.save_playback_file(self._playback_file)
            self._playback_file_save = False

    def load_playback_file(self, file_name: str) -> None:
        self._playback_file = file_name
        self._playback_file_load = True

    def _load_playback_file(self) -> None:
        if self._playback_file_load:
            self.playback_record_controller.load_playback_file(self._playback_file)
            self._playback_file_load = False

            self.update_ui_recorded_steps.emit(
                len(self.playback_record_controller.recorded_joint_positions),
                self.playback_record_controller.playback_index
            )
