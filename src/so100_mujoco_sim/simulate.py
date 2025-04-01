import mujoco
import numpy as np
import pathlib
import time

from collections import deque
from PySide6.QtCore import (
    QTimer, Qt, Signal, Slot, QThread, QSettings
)
from PySide6.QtGui import (
    QGuiApplication, QSurfaceFormat
)
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QPushButton, QSizePolicy,
    QVBoxLayout, QGroupBox, QHBoxLayout, QSlider, QLabel, QFileDialog,
    QLineEdit, QLayout
)

from so100_mujoco_sim.arm_control import (
    joints_from_model,
    Joint,
    MujocoArmController
)


format = QSurfaceFormat()
format.setDepthBufferSize(24)
format.setStencilBufferSize(8)
format.setSamples(4)
format.setSwapInterval(1)
format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
format.setVersion(2,0)
format.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
format.setProfile(QSurfaceFormat.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(format)


class Viewport(QOpenGLWindow):

    updateRuntime = Signal(float)

    def __init__(self, model, data, cam, opt, scn) -> None:
        super().__init__()

        self.model = model
        self.data = data
        self.cam = cam
        self.opt = opt
        self.scn = scn

        self.width = 0
        self.height = 0
        self.scale = 1.0
        self.__last_pos = None

        self.runtime = deque(maxlen=1000)
        self.timer = QTimer()
        self.timer.setInterval(1/60*1000)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def mousePressEvent(self, event):
        self.__last_pos = event.position()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        pos = event.position()
        dx = pos.x() - self.__last_pos.x()
        dy = pos.y() - self.__last_pos.y()
        mujoco.mjv_moveCamera(self.model, action, dx / self.height, dy / self.height, self.scn, self.cam)
        self.__last_pos = pos

    def wheelEvent(self, event):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.0005 * event.angleDelta().y(), self.scn, self.cam)

    def initializeGL(self):
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)

    def resizeGL(self, w, h):
        self.width = w
        self.height = h

    def setScreenScale(self, scaleFactor: float) -> None:
        """ Sets a scale factor that is used to scale the OpenGL window to accommodate
        the high DPI scaling Qt does.
        """
        self.scale = scaleFactor

    def paintGL(self) -> None:
        t = time.time()
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        viewport = mujoco.MjrRect(0, 0, int(self.width * self.scale), int(self.height * self.scale))
        mujoco.mjr_render(viewport, self.scn, self.con)

        self.runtime.append(time.time()-t)
        self.updateRuntime.emit(np.average(self.runtime))


class UpdateSimThread(QThread):

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.mujoco_controller = MujocoArmController(model, data)
        self.running = True

        self.mujoco_controller.reset()

        # reset the simulation timer
        self.reset()

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def run(self) -> None:
        while self.running:
            # don't step the simulation past real time
            # without this the sim usually finishes before it's
            # even visible
            if self.data.time < self.real_time / 1_000_000_000:
                # Update the control loop at a 100hz
                if (time.monotonic_ns() - self.last_robot_update) / 1_000_000_000 >= (1/100):
                    self.last_robot_update = time.monotonic_ns()
                    # apply the positions set via the UI to the mujoco model
                    self.mujoco_controller.set_positions()

                # step the simulation
                mujoco.mj_step(self.model, self.data)
                self.mujoco_controller.update()
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
        self.mujoco_controller.set_joint_set_position(joint_name, position)


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

        self.th = UpdateSimThread(self.model, self.data, self)
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

    def create_right_side_control(self) -> QLayout:

        # Add the Config group box
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)

        # Calibration file selection
        calibration_layout_v = QVBoxLayout()
        calibration_layout_v.setSpacing(0)
        calibration_layout_v.addWidget(QLabel("LeRobot Calibration File:"))
        calibration_layout = QHBoxLayout()
        calibration_layout.setSpacing(4)
        self.calibration_file_edit = QLineEdit()
        self.calibration_file_edit.setPlaceholderText("Select file...")
        calibration_button = QPushButton("Browse")
        calibration_button.clicked.connect(self._select_calibration_file)
        calibration_layout.addWidget(self.calibration_file_edit)
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
        
        control_layout = QVBoxLayout()
        # Add the Robot Control group box
        for joint in self.joints:
            widget = JointWidget(joint)
            widget.joint_position_changed.connect(self._joint_position_changed)
            control_layout.addWidget(widget)
            self.joint_widgets.append(widget)
        control_layout.addStretch()

        reset_button = QPushButton("Reset")
        reset_button.setMinimumWidth(90)
        reset_button.clicked.connect(self.reset_simulation)
        control_layout.addWidget(reset_button)

        robot_control_group = QGroupBox("Robot Control")
        robot_control_group.setLayout(control_layout)
        robot_control_group.setMinimumWidth(300)

        layout = QVBoxLayout()
        layout.setSpacing(24)
        layout.addWidget(config_group)
        layout.addWidget(robot_control_group)

        return layout

    def _select_calibration_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Calibration File", "", "All Files (*)")
        if file_path:
            self.calibration_file_edit.setText(file_path)

    def restore_settings(self):
        """Restore saved settings for calibration file and USB port."""
        calibration_file = self.settings.value("calibration_file", "")
        usb_port = self.settings.value("usb_port", "")

        self.calibration_file_edit.setText(calibration_file)
        self.usb_port_edit.setText(usb_port)

    def closeEvent(self, event):
        """Save settings when the application is closed."""
        self.settings.setValue("calibration_file", self.calibration_file_edit.text())
        self.settings.setValue("usb_port", self.usb_port_edit.text())
        super().closeEvent(event)

    def _joint_position_changed(self, joint: Joint, position: float) -> None:
        self.th.set_joint_position(joint.name, position)

    def _speed_changed(self, value: int) -> None:
        speed = value / 1000
        self.th.set_speed(speed)

    def _yaw_changed(self, value: int) -> None:
        yaw = value / 1000
        self.th.set_yaw(yaw)

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
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
