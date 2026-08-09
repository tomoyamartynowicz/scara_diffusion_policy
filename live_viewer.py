import matplotlib.pyplot as plt
import numpy as np


# XY joint origins from precise-automation/tcs-ros-rviz/urdf/PF3400SX.urdf (metres).
SHOULDER = np.array([0.03005, -0.0198]) + np.array([0.173001, 0.020])
LINK_1 = 0.302
LINK_2 = 0.289
TOOL = 0.078573


def _link(length, angle):
    return length * np.array([np.cos(angle), np.sin(angle)])


def joint_points(q):
    """Return the shoulder, elbow, wrist and tool in the XY plane."""
    _, j2, j3, j4 = np.asarray(q, dtype=float)
    elbow = SHOULDER + _link(LINK_1, j2)
    wrist = elbow + _link(LINK_2, j2 + j3)
    tool = wrist + _link(TOOL, j2 + j3 + j4)
    return np.stack([SHOULDER, elbow, wrist, tool])


class LiveViewer:
    def __init__(self, show_camera=True):
        plt.ion()
        self.show_camera = show_camera
        self.figure = plt.figure("SCARA Diffusion Policy live viewer", figsize=(12, 6))
        self.robot_ax = self.figure.add_subplot(121)
        self.camera_ax = self.figure.add_subplot(122)
        self.camera_artist = self.camera_ax.imshow(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        self.camera_ax.axis("off")
        self.camera_ax.set_title("Policy camera — druk C om te toggelen")
        self.camera_ax.set_visible(show_camera)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        plt.show(block=False)

    def _on_key(self, event):
        if event.key and event.key.lower() == "c":
            self.show_camera = not self.show_camera
            self.camera_ax.set_visible(self.show_camera)
            self.figure.canvas.draw_idle()

    def update(self, image, current_q, predicted_q, chunk_number):
        if not plt.fignum_exists(self.figure.number):
            return False

        self.robot_ax.clear()
        current = joint_points(current_q)
        predicted = [joint_points(q) for q in predicted_q]

        for points in predicted:
            self.robot_ax.plot(*points.T, color="tab:orange", alpha=0.10, linewidth=1)
        if predicted:
            tools = np.array([points[-1] for points in predicted])
            self.robot_ax.plot(
                *tools.T, color="tab:red", linewidth=2, label="voorspeld toolpad"
            )
            self.robot_ax.plot(
                *predicted[-1].T,
                "o-",
                color="tab:orange",
                linewidth=2,
                label="laatste actie",
            )
        self.robot_ax.plot(
            *current.T, "o-", color="tab:blue", linewidth=3, label="huidige pose"
        )

        final_j1 = predicted_q[-1][0] if len(predicted_q) else current_q[0]
        self.robot_ax.set(
            xlim=(-0.75, 0.95),
            ylim=(-0.85, 0.85),
            xlabel="X [m]",
            ylabel="Y [m]",
            title=(
                f"Chunk {chunk_number}: {len(predicted)} acties | "
                f"J1 {current_q[0]:.3f} → {final_j1:.3f} m"
            ),
        )
        self.robot_ax.set_aspect("equal")
        self.robot_ax.grid()
        self.robot_ax.legend(loc="upper left")
        self.camera_artist.set_data(image)
        self.figure.canvas.draw_idle()
        plt.pause(0.001)
        return plt.fignum_exists(self.figure.number)

    def close(self):
        plt.close(self.figure)
