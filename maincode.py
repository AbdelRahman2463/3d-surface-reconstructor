import sys
import os
import open3d as o3d
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLineEdit, QFileDialog, QLabel, QSlider, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QWindow

class ModernReconstructionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.pcd = None
        self.mesh = None
        self.vis = None
        self.init_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_vis_window)
        self.timer.start(16)

        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.update_reconstruction)

    def init_ui(self):
        self.setWindowTitle("3D Surface Reconstructor Dashboard")
        self.resize(1100, 650)
        
        main_layout = QHBoxLayout()
        
        control_panel = QWidget()
        control_panel.setFixedWidth(360)
        control_layout = QVBoxLayout(control_panel)
        
        file_layout = QHBoxLayout()
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select input point cloud (.ply, .pcd)...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.browse_btn)
        control_layout.addLayout(file_layout)
        
        self.method_label = QLabel("Reconstruction Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Poisson Surface Reconstruction", 
            "Ball Pivoting Algorithm (BPA)",
            "B-Spline / NURBS Surface (Smooth Sheet)",
            "Polynomial Surface Fitting"
        ])
        self.method_combo.currentIndexChanged.connect(self.on_method_changed)
        control_layout.addWidget(self.method_label)
        control_layout.addWidget(self.method_combo)
        
        # --- COMMON SLIDERS ---
        self.knn_label = QLabel("Normal Estimation Neighbors (KNN): 50")
        self.knn_slider = QSlider(Qt.Orientation.Horizontal)
        self.knn_slider.setMinimum(5)
        self.knn_slider.setMaximum(80)
        self.knn_slider.setValue(50)  # Poisson default setting start
        self.knn_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.knn_label)
        control_layout.addWidget(self.knn_slider)
        
        # --- POISSON SPECIFIC SLIDERS ---
        self.depth_label = QLabel("Octree Depth (Resolution): 5")
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setMinimum(5)  
        self.depth_slider.setMaximum(10)  
        self.depth_slider.setValue(5)
        self.depth_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.depth_label)
        control_layout.addWidget(self.depth_slider)
        
        self.trim_label = QLabel("Artifact Removal Quantile: 0.00")
        self.trim_slider = QSlider(Qt.Orientation.Horizontal)
        self.trim_slider.setMinimum(0)
        self.trim_slider.setMaximum(10)
        self.trim_slider.setValue(0)
        self.trim_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.trim_label)
        control_layout.addWidget(self.trim_slider)
        
        # --- BALL PIVOTING SPECIFIC SLIDERS ---
        self.voxel_label = QLabel("Grid Uniformity Filter (Voxel Size): 3.0x")
        self.voxel_slider = QSlider(Qt.Orientation.Horizontal)
        self.voxel_slider.setMinimum(0)   
        self.voxel_slider.setMaximum(30)  
        self.voxel_slider.setValue(30) # Maps to 3.0x layout step
        self.voxel_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.voxel_label)
        control_layout.addWidget(self.voxel_slider)

        self.bpa_label = QLabel("Ball Radius Scale Multiplier: 5.5")
        self.bpa_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpa_slider.setMinimum(2)    
        self.bpa_slider.setMaximum(60)   
        self.bpa_slider.setValue(55) # Maps to 5.5x layout step
        self.bpa_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.bpa_label)
        control_layout.addWidget(self.bpa_slider)

        # --- GRID & SMOOTHNESS (SHARED BY B-SPLINE AND POLYNOMIAL) ---
        self.grid_res_label = QLabel("Control Grid Resolution: 40 x 40")
        self.grid_res_slider = QSlider(Qt.Orientation.Horizontal)
        self.grid_res_slider.setMinimum(10)
        self.grid_res_slider.setMaximum(120)
        self.grid_res_slider.setValue(40)
        self.grid_res_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.grid_res_label)
        control_layout.addWidget(self.grid_res_slider)

        self.spline_smooth_label = QLabel("Spline Smoothing Factor: 1.0")
        self.spline_smooth_slider = QSlider(Qt.Orientation.Horizontal)
        self.spline_smooth_slider.setMinimum(1)
        self.spline_smooth_slider.setMaximum(100)
        self.spline_smooth_slider.setValue(10)
        self.spline_smooth_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.spline_smooth_label)
        control_layout.addWidget(self.spline_smooth_slider)

        # --- POLYNOMIAL SPECIFIC SLIDERS ---
        self.poly_degree_label = QLabel("Polynomial Order/Degree: 2 (Quadratic)")
        self.poly_degree_slider = QSlider(Qt.Orientation.Horizontal)
        self.poly_degree_slider.setMinimum(1)
        self.poly_degree_slider.setMaximum(5)
        self.poly_degree_slider.setValue(2)
        self.poly_degree_slider.valueChanged.connect(self.on_slider_moved)
        control_layout.addWidget(self.poly_degree_label)
        control_layout.addWidget(self.poly_degree_slider)
        
        # --- VIEWPORT VIEW OPTIONS ---
        self.features_group_label = QLabel("<b>Viewport View Options:</b>")
        self.features_group_label.setStyleSheet("margin-top: 10px; color: #2b78e4;")
        control_layout.addWidget(self.features_group_label)

        self.show_points_cb = QCheckBox("Overlay Raw Point Features")
        self.show_points_cb.stateChanged.connect(self.on_feature_toggled)
        control_layout.addWidget(self.show_points_cb)

        # --- LIVE POINT DETAILS READOUT (Triangles Line Removed) ---
        self.details_box = QLabel("<b>Model Metrics Readout:</b><br>Points: 0")
        self.details_box.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 6px; margin-top: 5px;")
        control_layout.addWidget(self.details_box)
        
        # Hide non-Poisson options at start
        self.voxel_label.hide()
        self.voxel_slider.hide()
        self.bpa_label.hide()
        self.bpa_slider.hide()
        self.grid_res_label.hide()
        self.grid_res_slider.hide()
        self.spline_smooth_label.hide()
        self.spline_smooth_slider.hide()
        self.poly_degree_label.hide()
        self.poly_degree_slider.hide()
        
        self.save_btn = QPushButton("Export Reconstructed Model...")
        self.save_btn.setStyleSheet("font-weight: bold; height: 35px; background-color: #2b78e4; color: white; border-radius: 4px; margin-top: 10px;")
        self.save_btn.clicked.connect(self.export_mesh)
        control_layout.addWidget(self.save_btn)
        
        self.status_label = QLabel("Status: Please load a point cloud file.")
        self.status_label.setStyleSheet("color: gray;")
        control_layout.addStretch()
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_panel)
        
        self.view_placeholder = QWidget()
        self.placeholder_layout = QVBoxLayout(self.view_placeholder)
        self.placeholder_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.view_placeholder, stretch=1)
        
        self.setLayout(main_layout)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Point Cloud", "", "Point Cloud Files (*.ply *.pcd *.xyz)"
        )
        if file_path:
            self.file_input.setText(file_path)
            self.status_label.setText("Status: Loading file...")
            QApplication.processEvents()
            
            self.mesh = None
            self.pcd = o3d.io.read_point_cloud(file_path)
            self.status_label.setText(f"Status: Loaded {len(self.pcd.points)} points. Creating visualizer...")
            
            if self.vis is None:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(window_name="Live 3D Viewport", width=600, height=500, visible=False)
                
                if sys.platform == "win32":
                    import win32gui
                    import time
                    time.sleep(0.05)
                    hwnd = win32gui.FindWindowEx(0, 0, None, "Live 3D Viewport")
                    if hwnd:
                        o3d_win = QWindow.fromWinId(hwnd)
                        embedded_widget = QWidget.createWindowContainer(o3d_win, self.view_placeholder)
                        self.placeholder_layout.addWidget(embedded_widget)
                
                opt = self.vis.get_render_option()
                opt.mesh_show_wireframe = False
                opt.mesh_show_back_face = True
                opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
            
            self.update_reconstruction()

    def on_method_changed(self):
        method_idx = self.method_combo.currentIndex()
        
        # Block signals temporarily to prevent multiple redundant background processing loops while updating slider state baselines
        self.knn_slider.blockSignals(True)
        self.depth_slider.blockSignals(True)
        self.trim_slider.blockSignals(True)
        self.voxel_slider.blockSignals(True)
        self.bpa_slider.blockSignals(True)
        self.grid_res_slider.blockSignals(True)
        self.spline_smooth_slider.blockSignals(True)
        self.poly_degree_slider.blockSignals(True)

        # Inject default method settings dynamically
        if method_idx == 0:  # Poisson
            self.knn_slider.setValue(50)
            self.depth_slider.setValue(5)
            self.trim_slider.setValue(0)
        elif method_idx == 1:  # Ball Pivoting
            self.knn_slider.setValue(55)
            self.voxel_slider.setValue(30) # 3.0x
            self.bpa_slider.setValue(55)   # 5.5x
        elif method_idx == 2:  # B-Spline
            self.grid_res_slider.setValue(47)
            self.spline_smooth_slider.setValue(8) # 0.8 factor
        elif method_idx == 3:  # Polynomial
            self.grid_res_slider.setValue(47)
            self.poly_degree_slider.setValue(5)

        # Re-trigger visible string updates manually
        self.on_slider_moved()

        # Unblock slider signals
        self.knn_slider.blockSignals(False)
        self.depth_slider.blockSignals(False)
        self.trim_slider.blockSignals(False)
        self.voxel_slider.blockSignals(False)
        self.bpa_slider.blockSignals(False)
        self.grid_res_slider.blockSignals(False)
        self.spline_smooth_slider.blockSignals(False)
        self.poly_degree_slider.blockSignals(False)

        # Toggle display visibility layouts
        self.depth_label.setVisible(method_idx == 0)
        self.depth_slider.setVisible(method_idx == 0)
        self.trim_label.setVisible(method_idx == 0)
        self.trim_slider.setVisible(method_idx == 0)
        
        self.voxel_label.setVisible(method_idx == 1)
        self.voxel_slider.setVisible(method_idx == 1)
        self.bpa_label.setVisible(method_idx == 1)
        self.bpa_slider.setVisible(method_idx == 1)

        self.grid_res_label.setVisible(method_idx in [2, 3])
        self.grid_res_slider.setVisible(method_idx in [2, 3])
        
        self.spline_smooth_label.setVisible(method_idx == 2)
        self.spline_smooth_slider.setVisible(method_idx == 2)
        
        self.poly_degree_label.setVisible(method_idx == 3)
        self.poly_degree_slider.setVisible(method_idx == 3)
        
        self.knn_label.setVisible(method_idx in [0, 1])
        self.knn_slider.setVisible(method_idx in [0, 1])
        
        if self.pcd is not None:
            self.status_label.setText("Status: Method changed...")
            self.debounce_timer.start(250)

    def on_slider_moved(self):
        self.knn_label.setText(f"Normal Estimation Neighbors (KNN): {self.knn_slider.value()}")
        self.depth_label.setText(f"Octree Depth (Resolution): {self.depth_slider.value()}")
        self.trim_label.setText(f"Artifact Removal Quantile: {self.trim_slider.value() / 100:.2f}")
        
        v_val = self.voxel_slider.value()
        self.voxel_label.setText(f"Grid Uniformity Filter (Voxel Size): {v_val / 10:.1f}x" if v_val > 0 else "Grid Uniformity Filter (Voxel Size): Off")
        self.bpa_label.setText(f"Ball Radius Scale Multiplier: {self.bpa_slider.value() / 10:.1f}")
        
        grid_res = self.grid_res_slider.value()
        self.grid_res_label.setText(f"Control Grid Resolution: {grid_res} x {grid_res}")
        self.spline_smooth_label.setText(f"Spline Smoothing Factor: {self.spline_smooth_slider.value() / 10:.1f}")
        
        deg_names = {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic", 5: "Quintic"}
        deg_val = self.poly_degree_slider.value()
        self.poly_degree_label.setText(f"Polynomial Order/Degree: {deg_val} ({deg_names[deg_val]})")
        
        if self.pcd is not None and not self.knn_slider.signalsBlocked():
            self.status_label.setText("Status: Tweaking parameters...")
            self.debounce_timer.start(250)

    def on_feature_toggled(self):
        if self.pcd is not None:
            self.update_reconstruction()

    def update_reconstruction(self):
        if self.pcd is None:
            return
            
        self.status_label.setText("Status: Processing Surface Reconstruction...")
        QApplication.processEvents()
        
        method_idx = self.method_combo.currentIndex()
        
        knn = self.knn_slider.value() if method_idx in [0, 1] else 30
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        self.pcd.orient_normals_consistent_tangent_plane(k=knn)

        if method_idx == 0:
            # --- POISSON SURFACE RECONSTRUCTION ---
            depth = self.depth_slider.value()
            quantile = self.trim_slider.value() / 100.0
            
            raw_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=depth)
            densities = np.asarray(densities)
            cutoff = np.quantile(densities, quantile)
            raw_mesh.remove_vertices_by_mask(densities < cutoff)
            
        elif method_idx == 1:
            # --- BALL PIVOTING ALGORITHM (BPA) ---
            distances = self.pcd.compute_nearest_neighbor_distance()
            gap_baseline = np.percentile(distances, 95) if len(distances) > 0 else 0.1
            
            working_pcd = self.pcd
            if self.voxel_slider.value() > 0:
                voxel_size = gap_baseline * (self.voxel_slider.value() / 10.0)
                working_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
                distances = working_pcd.compute_nearest_neighbor_distance()
                gap_baseline = np.percentile(distances, 95) if len(distances) > 0 else 0.1

            working_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
            working_pcd.orient_normals_consistent_tangent_plane(k=knn)
            
            multiplier = self.bpa_slider.value() / 10.0
            base_radius = gap_baseline * multiplier
            
            radii = o3d.utility.DoubleVector([
                base_radius * 1.0, base_radius * 1.5, base_radius * 2.5, base_radius * 4.0
            ])
            raw_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(working_pcd, radii)
            
        elif method_idx == 2:
            # --- B-SPLINE / NURBS SURFACE FITTING ---
            try:
                from scipy.interpolate import SmoothBivariateSpline
            except ImportError:
                self.status_label.setText("Status: Error! Run 'pip install scipy'")
                return

            pts = np.asarray(self.pcd.points)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

            grid_res = self.grid_res_slider.value()
            s_factor = (self.spline_smooth_slider.value() / 10.0) * len(x)

            spline = SmoothBivariateSpline(x, y, z, s=s_factor)

            grid_x = np.linspace(x.min(), x.max(), grid_res)
            grid_y = np.linspace(y.min(), y.max(), grid_res)
            gx, gy = np.meshgrid(grid_x, grid_y)
            gz = spline.ev(gx, gy)

            vertices = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
            triangles = []
            for i in range(grid_res - 1):
                for j in range(grid_res - 1):
                    v0 = i * grid_res + j
                    v1 = v0 + 1
                    v2 = (i + 1) * grid_res + j
                    v3 = v2 + 1
                    triangles.append([v0, v1, v2])
                    triangles.append([v1, v3, v2])
            
            raw_mesh = o3d.geometry.TriangleMesh()
            raw_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            raw_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

        else:
            # --- POLYNOMIAL SURFACE FITTING ---
            pts = np.asarray(self.pcd.points)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            
            degree = self.poly_degree_slider.value()
            grid_res = self.grid_res_slider.value()
            
            A = []
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    A.append((x ** i) * (y ** j))
            A = np.vstack(A).T
            
            coefficients, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            
            grid_x = np.linspace(x.min(), x.max(), grid_res)
            grid_y = np.linspace(y.min(), y.max(), grid_res)
            gx, gy = np.meshgrid(grid_x, grid_y)
            
            flat_x, flat_y = gx.ravel(), gy.ravel()
            flat_z = np.zeros_like(flat_x)
            
            idx = 0
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    flat_z += coefficients[idx] * (flat_x ** i) * (flat_y ** j)
                    idx += 1
                    
            vertices = np.vstack([flat_x, flat_y, flat_z]).T
            triangles = []
            for i in range(grid_res - 1):
                for j in range(grid_res - 1):
                    v0 = i * grid_res + j
                    v1 = v0 + 1
                    v2 = (i + 1) * grid_res + j
                    v3 = v2 + 1
                    triangles.append([v0, v1, v2])
                    triangles.append([v1, v3, v2])
                    
            raw_mesh = o3d.geometry.TriangleMesh()
            raw_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            raw_mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))

        # Common cleanup
        raw_mesh.remove_degenerate_triangles()
        raw_mesh.remove_duplicated_triangles()
        raw_mesh.remove_duplicated_vertices()
        raw_mesh.remove_unreferenced_vertices()
        
        raw_mesh.compute_vertex_normals()
        raw_mesh.paint_uniform_color([0.1, 0.3, 0.9])
        
        # --- FIXED READOUT: ONLY POINTS LISTED ---
        num_pts = len(self.pcd.points)
        self.details_box.setText(f"<b>Model Metrics Readout:</b><br>Points: {num_pts:,}")

        # Camera view retention check
        is_subsequent_update = (self.mesh is not None)
        if is_subsequent_update:
            view_control = self.vis.get_view_control()
            camera_params = view_control.convert_to_pinhole_camera_parameters()
        
        self.vis.clear_geometries()
        self.mesh = raw_mesh
        
        self.vis.add_geometry(self.mesh)
        
        if self.show_points_cb.isChecked():
            self.vis.add_geometry(self.pcd)

        # Removed normal spikes reference rendering option block
        opt = self.vis.get_render_option()
        opt.point_show_normal = False
        
        if is_subsequent_update:
            view_control = self.vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        else:
            self.vis.reset_view_point(True)
            
        self.vis.update_renderer()
        self.status_label.setText(f"Status: Done rendering ({len(raw_mesh.triangles)} triangles)")

    def update_vis_window(self):
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()

    def export_mesh(self):
        if self.mesh is None:
            self.status_label.setText("Status: No model generated to save yet!")
            return
            
        file_filter = "Wavefront OBJ (*.obj);;Stereolithography STL (*.stl);;STEP Standard CAD (*.step)"
        output_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Reconstructed Mesh", "reconstructed_model.obj", file_filter
        )
        
        if not output_path:
            return 
            
        ext = os.path.splitext(output_path)[1].lower()
        self.status_label.setText(f"Status: Exporting file as {ext}...")
        QApplication.processEvents()
        
        try:
            if ext == ".step":
                try:
                    import trimesh
                except ImportError:
                    self.status_label.setText("Status: Error! Run 'pip install trimesh' for STEP export.")
                    return
                verts = np.asarray(self.mesh.vertices)
                faces = np.asarray(self.mesh.triangles)
                tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                tri_mesh.export(output_path)
            else:
                o3d.io.write_triangle_mesh(output_path, self.mesh)
                
            self.status_label.setText(f"Status: Successfully saved to {os.path.basename(output_path)}!")
        except Exception as e:
            self.status_label.setText(f"Status: Export failed! {str(e)}")

    def closeEvent(self, event):
        if self.vis is not None:
            self.vis.destroy_window()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernReconstructionApp()
    window.show()
    sys.exit(app.exec())