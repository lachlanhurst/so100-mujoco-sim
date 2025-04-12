import pathlib

import mujoco
import numpy as np
import qtawesome as qta
from PySide6.QtCore import QSettings, Qt, Signal, Slot, QSize
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QMainWindow, QMessageBox, QPushButton,
    QSlider, QVBoxLayout, QWidget
)

from so100_mujoco_sim.arm_control import (
    Joint,
    joints_from_model
)
from so100_mujoco_sim.mujoco_viewport import Viewport
from so100_mujoco_sim.update_thread import UpdateThread


class JointWidget(QWidget):
    """
    Wraps up the joint name and slider in a single widget.
    Two position values are displayed, the set position and
    the actual position from the simulation.
    """

    joint_position_changed = Signal(Joint, float)

    def __init__(self, joint: Joint) -> None:
        super().__init__()
        self.joint = joint
        self.actual_position: float = 0.0
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        name_label = QLabel(self.joint.name)
        self.actual_value_label = QLabel("0.0")
        self.actual_value_label.setStyleSheet("color: #888")
        self.actual_value_label.setMinimumWidth(40)
        self.value_label = QLabel("0.0")
        self.value_label.setStyleSheet("color: #cccccc")
        self.value_label.setMinimumWidth(40)
        label_layout = QHBoxLayout()
        label_layout.addWidget(name_label)
        label_layout.addStretch()
        
        label_layout.addWidget(self.value_label)
        label_layout.addWidget(self.actual_value_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.joint.range[0] * 1000)
        self.slider.setMaximum(self.joint.range[1] * 1000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._changed)
        layout.addLayout(label_layout)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def setValue(self, val: float):
        self.slider.setValue(val * 1000)

    def _changed(self, value: int) -> None:
        self.value_label.setText("{:.2f}".format(value / 1000.0))
        self.joint_position_changed.emit(self.joint, value / 1000.0)

    def set_actual_position(self, position: float) -> None:
        self.actual_value_label.setText("{:.2f}".format(position))


class Window(QMainWindow):
    """
    Main window for the application
    """

    def __init__(self) -> None:
        super().__init__()

        # Initialize QSettings for saving and restoring values
        self.settings = QSettings("lh", "So100MujocoSim")

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('xml/sim_scene.xml')))
        self.joints = joints_from_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())
        self.viewport.updateRuntime.connect(self.show_runtime)

        layout = QHBoxLayout()
        layout.setSpacing(0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.th = UpdateThread(self.model, self.data, self)
        self.th.update_ui_joint_values.connect(self._update_ui_joint_values)
        self.th.update_controller_enabled_states.connect(self._update_controllers_enabled)
        self.th.update_primary_controller.connect(self._primary_controller_changed)

        layout_right_side = QVBoxLayout()
        layout_right_side.setSpacing(8)
        layout_robot_controls = QVBoxLayout()
        self.joint_widgets: list[JointWidget] = []
        layout_robot_controls.addLayout(self.create_right_side_control())
        layout_right_side.addLayout(layout_robot_controls)
        layout_right_side.setContentsMargins(8,8,8,8)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.addLayout(layout_right_side)
        layout.setContentsMargins(0,0,0,0)
        layout.setStretch(0,1)

        self.resize(900, 600)

        self.th.start()

        # Restore saved settings
        self.restore_settings()

    @Slot(float)
    def show_runtime(self, fps: float):
        self.statusBar().showMessage(
            f"Average runtime: {fps:.0e}s\t"
            f"Simulation time: {self.data.time:.0f}s"
        )

        for i in range(len(self.joints)):
            pos = self.th.mujoco_controller.joint_actual_positions[i]
            jw = self.joint_widgets[i]
            jw.set_actual_position(pos)

    @Slot(list)
    def _update_ui_joint_values(self, joint_vals: list):
        # print("joint_vals")
        # print(joint_vals)

        for i, jw in enumerate(self.joint_widgets):
            jw.setValue(joint_vals[i])

    def _set_enable_ui_joint_controls(self, enabled: bool):
        for jw in self.joint_widgets:
            jw.slider.setEnabled(enabled)

    @Slot(str)
    def _primary_controller_changed(self, name: str):
        # disable or enable the joint controls depending on if the UI controller
        # is primary
        # should probably get this name from the controller itself
        self._set_enable_ui_joint_controls(name == "User Interface")

    def _create_config_group(self) -> QGroupBox:
        # Add the Config group box
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)

        # Calibration folder selection
        calibration_layout_v = QVBoxLayout()
        calibration_layout_v.setSpacing(0)
        calibration_layout_v.addWidget(QLabel("LeRobot Calibration Folder:"))
        calibration_layout = QHBoxLayout()
        calibration_layout.setSpacing(4)
        self.calibration_folder_edit = QLineEdit()
        self.calibration_folder_edit.setPlaceholderText("Select folder...")
        calibration_folder_open_icon = qta.icon("fa6.folder-open")
        calibration_button = QPushButton(calibration_folder_open_icon, "")
        calibration_button.clicked.connect(self._select_calibration_folder)
        calibration_layout.addWidget(self.calibration_folder_edit)
        calibration_layout.addWidget(calibration_button)
        calibration_layout_v.addLayout(calibration_layout)

        # USB port field
        usb_port_layout = QVBoxLayout()
        usb_port_layout.setSpacing(0)
        usb_port_label = QLabel("USB Port:")
        usb_port_layout.addWidget(usb_port_label)
        self.usb_port_edit = QLineEdit()
        self.usb_port_edit.setPlaceholderText("Enter USB port...")
        usb_port_layout.addWidget(self.usb_port_edit)
        config_layout.addLayout(calibration_layout_v)
        config_layout.addLayout(usb_port_layout)

        config_group = QGroupBox("Config")
        config_group.setLayout(config_layout)
        return config_group

    def _create_robot_control_group(self) -> QGroupBox:
        control_layout = QVBoxLayout()
        # Add the Robot Control group box
        for joint in self.joints:
            widget = JointWidget(joint)
            widget.joint_position_changed.connect(self._joint_position_changed)
            control_layout.addWidget(widget)
            self.joint_widgets.append(widget)
        control_layout.addStretch()

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_simulation)
        control_layout.addWidget(reset_button)

        robot_control_group = QGroupBox("Robot Control")
        robot_control_group.setLayout(control_layout)
        robot_control_group.setMinimumWidth(300)

        return robot_control_group

    def create_right_side_control(self) -> QLayout:
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect_robot)

        # Dropdown for selecting the primary controller
        controller_layout = QHBoxLayout()
        controller_layout.setSpacing(4)
        controller_layout.addWidget(QLabel("Control with:"))
        self.controller_dropdown = QComboBox()
        self.controller_dropdown.addItems(self.th.get_controller_names())
        self.controller_dropdown.setCurrentIndex(self.th.get_primary_controller_index())
        self.controller_dropdown.currentIndexChanged.connect(self._set_primary_controller)
        self._update_controllers_enabled()
        controller_layout.addWidget(self.controller_dropdown)
        controller_layout.setStretch(1,1)

        robot_control_layout = QVBoxLayout()
        robot_control_layout.setContentsMargins(0,0,0,0)
        robot_control_layout.addWidget(self._create_robot_control_group())

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.addWidget(self._create_config_group())
        layout.addWidget(self.connect_button)
        layout.addLayout(controller_layout)
        layout.addLayout(robot_control_layout)
        return layout

    def _update_controllers_enabled(self):
        controllable_controllers = self.th.get_controllable_controllers()
        for i, c in enumerate(controllable_controllers):
            self.controller_dropdown.model().item(i).setEnabled(c)
        
        for i, c in enumerate(self.th.get_controller_names()):
            if c == "Robot" and controllable_controllers[i]:
                # the robot controller is enabled, therefore the robot is connected
                # and we should disable the connect button (connecting twice breaks
                # things)
                self.connect_button.setText("Connected")
                self.connect_button.setEnabled(False)

    def _select_calibration_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if folder_path:
            self.calibration_folder_edit.setText(folder_path)

    def _set_primary_controller(self, index: int):
        """
        Sets the selected ArmController as the primary controller.
        """
        self.th.set_primary_controller_index(index)

    def restore_settings(self):
        """Restore saved settings for calibration folder and USB port."""
        calibration_folder = self.settings.value("calibration_folder", "")
        usb_port = self.settings.value("usb_port", "")

        self.calibration_folder_edit.setText(calibration_folder)
        self.usb_port_edit.setText(usb_port)

    def closeEvent(self, event):
        """Save settings when the application is closed."""
        self.settings.setValue("calibration_folder", self.calibration_folder_edit.text())
        self.settings.setValue("usb_port", self.usb_port_edit.text())
        super().closeEvent(event)

    def _joint_position_changed(self, joint: Joint, position: float) -> None:
        self.th.set_joint_position(joint.name, position)

    def show_warning_dialog(self, title: str, message: str) -> None:
        """
        Displays a warning dialog with the given title and message.

        :param title: The title of the warning dialog.
        :param message: The warning message to display.
        """
        warning_dialog = QMessageBox(self)
        warning_dialog.setIcon(QMessageBox.Icon.Warning)
        warning_dialog.setWindowTitle(title)
        warning_dialog.setText(message)
        warning_dialog.setStandardButtons(QMessageBox.Ok)
        warning_dialog.exec()

    def _connect_robot(self):
        calibration_folder = self.calibration_folder_edit.text()
        usb_port = self.usb_port_edit.text()

        if calibration_folder.strip() == "":
            self.show_warning_dialog(
                "Calibration File Error",
                "Please select a calibration file."
            )
            return
        if usb_port.strip() == "":
            self.show_warning_dialog(
                "USB Port Error",
                "Please enter a USB port."
            )
            return
        # Check if the calibration file exists
        if not pathlib.Path(calibration_folder).exists():
            self.show_warning_dialog(
                "Calibration File Error",
                f"Calibration file '{calibration_folder}' does not exist."
            )
            return

        self.th.connect_real_robot(usb_port, calibration_folder)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])
        cam.distance = self.model.stat.extent * 1.5
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        for jw in self.joint_widgets:
            jw.slider.setValue(0)
        # Reset state and time.
        mujoco.mj_resetData(self.model, self.data)
        self.th.reset()


if __name__ == "__main__":
    app = QApplication()
    app.setStyle('fusion')
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
