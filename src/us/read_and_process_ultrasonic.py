from scipy.ndimage import map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, buttord, hilbert
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sio
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QRect
from scipy.signal import correlate2d
from scipy.ndimage import zoom, gaussian_filter
from matplotlib.patches import Rectangle
from skimage.feature import match_template
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
from typing import Tuple
import plotly.graph_objects as go
from scipy.spatial import KDTree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def design_filter(f0,fs,bandwidth):
    nyq = 0.5 * fs
    lowcut = f0*(1-bandwidth)
    highcut = f0*(1+bandwidth)
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.001, min(0.999, low))
    high = max(0.001, min(0.999, high))
    # Ensure high > low
    if high <= low:
        high = min(0.999, low + 0.001)

    order = buttord([low, high],[low*(1-bandwidth), high*(1+bandwidth)],3,30)
    N = max(1, int(order[0]))
    b, a = butter(N, [low, high], btype='band')

    return b, a

class PostProcessor():
    def __init__(self,meta,**kwargs):

        self.meta = meta
        self.outImgSize = [1024,1024] # [pixels]
        self.Sequence  = meta.get('Sequence')
        self.f0 = meta.get('Sequence').get('f0')
        self.fs = meta.get('fs')
        self.bandwidth = 0.8
        self.imgRanges = meta.get('Beamforming').get('imgRanges')

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.b, self.a = design_filter(self.f0,self.fs,self.bandwidth)

    def upload_BfData(self,BfData):
        self.BfData = BfData

    def run(self):
        self.zero_pad()
        self.filterData()
        self.envelopeDetection()
        self.remove_pad()
        self.normaliseData()
        self.logCompression()
        PostDataset = self.scanConversion()
        return PostDataset

    def zero_pad(self):
        pad = np.int32(len(self.imgRanges)/4)
        temp = np.zeros((self.BfData.shape[0]+pad+pad,self.BfData.shape[1]), dtype=complex)
        temp[pad:len(self.imgRanges)+pad,:] = self.BfData
        self.BfData = temp

    def remove_pad(self):
        pad = np.int32(len(self.imgRanges)/4)
        self.BfData = self.BfData[pad:len(self.imgRanges)+pad,:]

    def filterData(self):
        self.BfData = filtfilt(self.b,self.a,self.BfData,axis=0)

    def envelopeDetection(self):
        if np.iscomplexobj(self.BfData) == True:
            self.BfData = abs(self.BfData)
        else:
            self.BfData = abs(hilbert(self.BfData,axis=0))

    def normaliseData(self):
        norm = np.max(self.BfData.flatten())
        self.BfData = self.BfData/norm

    def logCompression(self):
        self.BfData = np.nan_to_num(20*np.log10(self.BfData))

    def scanConversion(self):

        lat_min, lat_max = -19.2e-3, 19.2e-3

        x = np.linspace(0,1,self.outImgSize[0]) * (lat_max-lat_min) + lat_min
        z = np.linspace(0,1,self.outImgSize[1]) * (self.imgRanges[-1]-self.imgRanges[0]) + self.imgRanges[0]

        X, Z = np.meshgrid(x, z)

        ir = interp1d(self.imgRanges, np.arange(len(self.imgRanges)), bounds_error=False)
        nScanLines = self.Sequence.get('nScanLines')[0]

        lateral_positions = np.linspace(lat_min, lat_max, nScanLines)
        it = interp1d(lateral_positions, np.arange(nScanLines), bounds_error=False)

        new_ir = ir(Z.ravel())
        new_it = it(X.ravel())

        bMode = map_coordinates(self.BfData, np.array([new_ir, new_it]),order=1,mode='constant',cval=np.nan).reshape(X.shape)

        return {'bMode': bMode,'xAxis': x,'zAxis': z}

def plot_Bmode(PostDataset, **kwargs):

    bMode = PostDataset.get('bMode')
    xAxis = PostDataset.get('xAxis')*1000
    zAxis = PostDataset.get('zAxis')*1000
    if 'dRange' in kwargs:
        dRange = kwargs['dRange']
    else:
        dRange = -60

    if 'fontSize' in kwargs:
        fontSize = kwargs['fontSize']
    else:
        fontSize = 14

    latexFont='cmr10'
    fig = plt.figure()
    img = plt.imshow(bMode,
        interpolation = 'gaussian',
        cmap          = 'gray',
        aspect        = 'equal',
        extent        = (xAxis[0],xAxis[-1],zAxis[-1],zAxis[0]),
        clim          = (dRange,0))

    plt.xlabel('Distance from Centre of Array [mm]',fontsize=fontSize,fontname=latexFont)
    plt.ylabel('Depth [mm]',fontsize=fontSize,fontname=latexFont)
    plt.xticks(fontsize=fontSize,fontname=latexFont)
    plt.yticks(fontsize=fontSize,fontname=latexFont)
    plt.rc('axes', unicode_minus=False)
    plt.ylim([zAxis[-1],zAxis[0]])

    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=fontSize)
    cbar.set_label('Amplitude [dB]',rotation=90,labelpad=15,fontsize=fontSize,fontname=latexFont)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family(latexFont)
    plt.show()

class ImageSelector(QMainWindow):
    def __init__(self, image, x_range, y_range, title):
        super().__init__()
        self.image = image
        self.x_range = x_range
        self.y_range = y_range
        self.title = title
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        # Convert numpy array to QImage
        height, width = self.image.shape
        bytes_per_line = width

        # Apply the same colormap and normalization as in matplotlib
        normalized_image = np.clip((self.image - (-60)) / (0 - (-60)), 0, 1)
        colormap = plt.get_cmap('gray')
        rgba_img = (colormap(normalized_image) * 255).astype(np.uint8)

        qimage = QImage(rgba_img.data, width, height, QImage.Format_RGBA8888)

        self.label = QLabel(self)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

        self.button = QPushButton('Confirm Selection', self)
        self.button.clicked.connect(self.confirm_selection)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.begin = None
        self.end = None
        self.selection_made = False

    def paintEvent(self, event):
        if self.begin and self.end:
            painter = QPainter(self.label.pixmap())
            painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            painter.drawRect(QRect(self.begin, self.end))
            self.label.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.begin = event.pos() - self.label.pos()
            self.end = self.begin
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.end = event.pos() - self.label.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end = event.pos() - self.label.pos()
            self.selection_made = True
            self.update()

    def confirm_selection(self):
        if self.selection_made:
            self.close()
        else:
            print("No selection made. Please draw a rectangle before confirming.")

def process_image(image, x_range, y_range, title):
    app = QApplication(sys.argv)
    ex = ImageSelector(image, x_range, y_range, title)
    ex.show()
    app.exec_()

    if ex.selection_made:
        x1, y1 = ex.begin.x(), ex.begin.y()
        x2, y2 = ex.end.x(), ex.end.y()

        # Convert pixel coordinates to image coordinates
        height, width = image.shape
        x1 = int(x1 * width / ex.label.pixmap().width())
        x2 = int(x2 * width / ex.label.pixmap().width())
        y1 = int(y1 * height / ex.label.pixmap().height())
        y2 = int(y2 * height / ex.label.pixmap().height())

        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        current_template = image[y_min:y_max, x_min:x_max]
        return current_template
    else:
        print("No selection made. Returning None.")
        return None

def display_templates(templates, title):
    valid_templates = [t for t in templates if t is not None]
    if valid_templates:
        fig, axes = plt.subplots(1, len(valid_templates), figsize=(12, 6))
        if len(valid_templates) == 1:
            axes = [axes]
        for i, template in enumerate(valid_templates):
            axes[i].imshow(template, cmap='gray', vmin=-50, vmax=0)
            axes[i].set_title(f'{title} Template {i + 1}')
        plt.pause(0.001)

# Tracking
def normxcorr2(template, image):
    # Ensure inputs are 2D arrays
    if template.ndim != 2 or image.ndim != 2:
        raise ValueError('normxcorr2 inputs must both be 2-D arrays')
    # Normalized cross-correlation
    return correlate2d(image, template, mode='valid')

def imresize(image, new_size):
    # Simple resize function using scipy.ndimage.zoom
    zoom_factors = (new_size[0] / image.shape[0], new_size[1] / image.shape[1])
    return zoom(image, zoom_factors, order=1)

def get_current_template(templates):
    valid_templates = [t for t in templates if t is not None]
    if valid_templates:
        return np.mean(valid_templates, axis=0)
    return None

def preprocess_image(image):
    # Apply Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(image, sigma=1)

    # Enhance contrast
    p2, p98 = np.percentile(smoothed, (2, 98))
    enhanced = np.clip((smoothed - p2) / (p98 - p2), 0, 1)

    return enhanced

def template_match_multi_scale(image, template, scales):
    best_score = -np.inf
    best_match = None

    for scale in scales:
        resized = zoom(template, scale)
        if resized.shape[0] > image.shape[0] or resized.shape[1] > image.shape[1]:
            #print(f"Skipping scale {scale} as resized template is larger than image")
            continue
        result = match_template(image, resized)
        max_score = np.max(result)

        if max_score > best_score:
            best_score = max_score
            ij = np.unravel_index(np.argmax(result), result.shape)
            best_match = (ij[0], ij[1], scale)

    return best_match

def apply_template_matching(image, templates, templateIndex):
    current_template = get_current_template(templates)
    if current_template is not None:
        try:
            #print(f"Image shape: {image.shape}, Template shape: {current_template.shape}")

            # Preprocess the image and template
            preprocessed_image = preprocess_image(image)
            preprocessed_template = preprocess_image(current_template)

            #print(
                #f"Preprocessed image shape: {preprocessed_image.shape}, Preprocessed template shape: {preprocessed_template.shape}")

            # Ensure template is smaller than image
            if preprocessed_template.shape[0] >= preprocessed_image.shape[0] or preprocessed_template.shape[1] >= \
                    preprocessed_image.shape[1]:
                #print("Template is larger than or equal to image. Resizing template.")
                scale_factor = min(preprocessed_image.shape[0] / preprocessed_template.shape[0],
                                   preprocessed_image.shape[1] / preprocessed_template.shape[
                                       1]) * 0.9  # 90% of max possible size
                preprocessed_template = zoom(preprocessed_template, scale_factor)
                #print(f"Resized template shape: {preprocessed_template.shape}")

            # Perform multi-scale template matching
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
            match_result = template_match_multi_scale(preprocessed_image, preprocessed_template, scales)

            if match_result is None:
                print("No valid match found. Using fallback method.")
                # Fallback to simpler matching method
                result = match_template(preprocessed_image, preprocessed_template)
                ij = np.unravel_index(np.argmax(result), result.shape)
                yoffSet, xoffSet = ij
                best_scale = 1.0
            else:
                yoffSet, xoffSet, best_scale = match_result

            #print(f"Best match found at: ({xoffSet}, {yoffSet}) with scale {best_scale}")

            # Update template size based on best scale
            new_template_shape = tuple(int(s * best_scale) for s in current_template.shape)
            new_y_end = min(yoffSet + new_template_shape[0], image.shape[0])
            new_x_end = min(xoffSet + new_template_shape[1], image.shape[1])

            new_template = image[yoffSet:new_y_end, xoffSet:new_x_end]

            #print(f"New template shape: {new_template.shape}")

            # Ensure new template has the same shape as the current template
            if new_template.shape != current_template.shape:
                new_template = zoom(new_template, (current_template.shape[0] / new_template.shape[0],
                                                   current_template.shape[1] / new_template.shape[1]))

            #print(f"Final new template shape: {new_template.shape}")

            return (templateIndex + 1) % N, new_template, xoffSet, yoffSet
        except ValueError as e:
            print(f"Error during template matching: {e}")
    return templateIndex, None, None, None

def plot_image_with_rectangle(image, x, y, xoffSet, yoffSet, current_template, title):
    fig, ax = plt.subplots()
    ax.imshow(image, extent=[x[0], x[-1], y[-1], y[0]], cmap='gray', aspect='equal', vmin=-60, vmax=0)
    ax.set_title(title)

    # Convert pixel offsets to data coordinates
    x_start = x[xoffSet]
    y_start = y[yoffSet]
    x_end = x[min(xoffSet + current_template.shape[1] - 1, len(x) - 1)]
    y_end = y[min(yoffSet + current_template.shape[0] - 1, len(y) - 1)]

    width = x_end - x_start
    height = y_end - y_start

    rect = Rectangle((x_start, y_start), width, height, fill=False, edgecolor='r', linewidth=2)
    ax.add_patch(rect)

    plt.axis('tight')
    plt.show()

def Asy_icp(q: np.ndarray, p: np.ndarray, iter: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert q.shape[0] == 3 and p.shape[0] == 3, "Input point clouds must be 3xN arrays"
    assert 0 < iter < 1e5, "Number of iterations must be between 1 and 100,000"

    ER = np.zeros(iter + 1)
    TT = np.zeros((3, 1, iter + 1))
    TR = np.repeat(np.eye(3)[:, :, np.newaxis], iter + 1, axis=2)

    kdtree = KDTree(q.T)

    for k in range(iter):
        match, mindist = match_kDtree(q, p, kdtree)
        R, T = eq_point(q[:, match], p)

        TR[:, :, k + 1] = R @ TR[:, :, k]
        TT[:, :, k + 1] = R @ TT[:, :, k] + T[:, np.newaxis]

        if k == 0:
            ER[k] = np.sqrt(np.mean(mindist ** 2))

        p = R @ p + T[:, np.newaxis]
        ER[k + 1] = rms_error(q[:, match], p)

    return TR[:, :, -1], TT[:, :, -1], ER

def match_kDtree(q: np.ndarray, p: np.ndarray, kdtree: KDTree) -> Tuple[np.ndarray, np.ndarray]:

    mindist, match = kdtree.query(p.T)
    return match, mindist

def eq_point(q: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    q_bar = np.mean(q, axis=1)
    q_mark = q - q_bar[:, np.newaxis]
    p_bar = np.mean(p, axis=1)
    p_mark = p - p_bar[:, np.newaxis]

    N = p_mark @ q_mark.T
    U, _, Vt = np.linalg.svd(N)

    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    T = q_bar - R @ p_bar

    return R, T

def rms_error(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((p1 - p2) ** 2, axis=0)))
def reverseTransformation(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    R_inv = R.T
    T_inv = -R_inv @ T
    T_inv = T_inv.flatten()  # Ensure it's a 1D array

    return R_inv, T_inv

def polynomial_surface_smoothing(points, degree=4):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Prepare the features
    X = np.column_stack((x, y))
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Fit the model
    model = LinearRegression()
    model.fit(X_poly, z)

    # Create a grid of points
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Predict on the grid
    X_grid = np.column_stack((xi.ravel(), yi.ravel()))
    X_grid_poly = poly.transform(X_grid)
    zi = model.predict(X_grid_poly).reshape(xi.shape)

    # Interpolate back to original x, y coordinates
    smoothed_z = griddata((xi.ravel(), yi.ravel()), zi.ravel(), (x, y), method='linear')

    return np.column_stack((x, y, smoothed_z))
import plotly.graph_objects as go

def visualize_point_clouds(CT_transformed: np.ndarray, USptCloud: np.ndarray, frame_idx: int):
    """
    Visualizes the transformed CT point cloud and the US point cloud using Plotly.

    Parameters:
    - CT_transformed (np.ndarray): Transformed CT point cloud of shape (N, 3).
    - USptCloud (np.ndarray): US point cloud of shape (M, 3).
    - frame_idx (int): Current frame index for labeling.
    """
    # Initialize the figure
    fig = go.Figure()

    # Add CT Point Cloud
    fig.add_trace(go.Scatter3d(
        x=CT_transformed[:, 0],
        y=CT_transformed[:, 1],
        z=CT_transformed[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.8
        ),
        name='CT Point Cloud'
    ))

    # Add US Point Cloud
    fig.add_trace(go.Scatter3d(
        x=USptCloud[:, 0],
        y=USptCloud[:, 1],
        z=USptCloud[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.8
        ),
        name='US Point Cloud'
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f'Frame {frame_idx}: CT and US Point Clouds',
        scene=dict(
            xaxis_title='X [mm]',
            yaxis_title='Y [mm]',
            zaxis_title='Z [mm]',
            aspectmode='data',  # Ensures equal scaling for all axes
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust camera position as needed
            ),
            xaxis=dict(backgroundcolor="rgb(200, 200, 230)"),
            yaxis=dict(backgroundcolor="rgb(230, 200, 200)"),
            zaxis=dict(backgroundcolor="rgb(200, 230, 200)")
        ),
        legend=dict(
            x=0.8,
            y=0.9
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Display the figure
    fig.show()

def process_frame(frame_idx, BfMeta, transformed_CTptCloud, original_CTptCloud, left_templates, right_templates, left_templateIndex, right_templateIndex):

    # Load BfDataset
    filename = f'BfDataset_frame{frame_idx:03}.mat'
    BfDataset = sio.loadmat(path + filename, simplify_cells=True).get('BfDataset')

    # Post-processing
    PP = PostProcessor(BfMeta)
    PP.upload_BfData(BfDataset.get('BfData'))
    PostDataset = PP.run()

    # Split image and prepare for template matching
    midPoint = PostDataset['bMode'].shape[1] // 2
    leftimage = PostDataset['bMode'][:, :midPoint]
    rightimage = PostDataset['bMode'][:, midPoint:]
    x = PostDataset['xAxis'] * 1000  # Convert to mm
    y = PostDataset['zAxis'] * 1000  # Convert to mm

    # Template matching
    if frame_idx == 0:
        left_template = process_image(leftimage, x[:midPoint], y, '1st Array')
        right_template = process_image(rightimage, x[midPoint:], y, '2nd Array')
        left_templates = [left_template] * N
        right_templates = [right_template] * N
        left_templateIndex = 0
        right_templateIndex = 0
    else:
        left_templateIndex, new_left_template, left_xoffSet, left_yoffSet = apply_template_matching(leftimage,
                                                                                                    left_templates,
                                                                                                    left_templateIndex)
        right_templateIndex, new_right_template, right_xoffSet, right_yoffSet = apply_template_matching(rightimage,
                                                                                                        right_templates,
                                                                                                        right_templateIndex)
        if new_left_template is not None:
            left_templates[left_templateIndex] = new_left_template
            plot_image_with_rectangle(leftimage, x[:midPoint], y, left_xoffSet, left_yoffSet, new_left_template, f'Left Template Matching (Frame {frame_idx})')

        if new_right_template is not None:
            right_templates[right_templateIndex] = new_right_template
            plot_image_with_rectangle(rightimage, x[midPoint:], y, right_xoffSet, right_yoffSet, new_right_template, f'Right Template Matching (Frame {frame_idx})')

    # Generate point cloud
    Img1 = get_current_template(left_templates)
    Img2 = get_current_template(right_templates)
    binaryImg1 = (Img1 > -35).astype(float)
    binaryImg2 = (Img2 > -35).astype(float)
    x_vec = np.linspace(-7e-3, 7e-3, binaryImg1.shape[1])
    z_vec = np.linspace(11e-3, 15e-3, binaryImg1.shape[0])
    rows1, cols1 = np.nonzero(binaryImg1)
    rows2, cols2 = np.nonzero(binaryImg2)

    USset1 = np.column_stack((x_vec[cols1] * 1000, np.full(rows1.shape, 1), z_vec[rows1] * 1000))
    USset2 = np.column_stack(
        (np.take(x_vec, cols2, mode='clip') * 1000, np.full(rows2.shape, 4), z_vec[rows2] * 1000))

    x_data = np.concatenate((USset1[:, 0], USset2[:, 0]))
    y_data = np.concatenate((USset1[:, 1], USset2[:, 1]))
    z_data = np.concatenate((USset1[:, 2], USset2[:, 2]))

    # Interpolation
    x_common = np.unique(x_data)
    num_planes = 20
    y_planes = np.linspace(USset1[0, 1], USset2[0, 1], num_planes)

    USptCloud = []
    for y_plane in y_planes:
        X, Z = np.meshgrid(x_common, np.linspace(np.min(z_data), np.max(z_data), x_common.size))
        Z_interp = griddata((x_data, y_data), z_data, (X, np.full_like(X, y_plane)), method='linear')
        plane_ptCloud = np.column_stack((X.flatten(), np.full_like(X.flatten(), y_plane), Z_interp.flatten()))
        plane_ptCloud = plane_ptCloud[~np.isnan(plane_ptCloud).any(axis=1)]
        USptCloud.append(plane_ptCloud)

    USptCloud = np.unique(np.vstack(USptCloud), axis=0)

    smoothed_USptCloud = polynomial_surface_smoothing(USptCloud, degree=5)

    R, T, ER = Asy_icp(transformed_CTptCloud.T, smoothed_USptCloud.T, 100)
    R_inv, T_inv = reverseTransformation(R, T)

    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_inv
    transformation_matrix[:3, 3] = T_inv.flatten()

    return transformation_matrix, left_templates, right_templates, left_templateIndex, right_templateIndex, transformed_CTptCloud, smoothed_USptCloud

if __name__ == '__main__':
    path = r'C:\MINE\Python\\'
    nFrames = 1
    N = 20  # Number of templates to keep

    # Load CT point cloud
    #mat_contents = sio.loadmat('Arm_ph_vertices.mat')
    #CTptCloud = mat_contents['v']

    def stlread(filename):
        with open(filename, 'rb') as f:
            M = f.read()
        if is_binary_stl(M):
            return read_binary_stl(M)
        else:
            return read_ascii_stl(filename)

    def is_binary_stl(M):
        header = M[:80].decode('utf-8', errors='ignore')
        return 'solid' not in header

    def read_binary_stl(M):
        n_faces = np.frombuffer(M[80:84], dtype=np.uint32)[0]
        data = np.frombuffer(M[84:], dtype=np.uint8).reshape((-1, 50))
        n = np.frombuffer(data[:, 0:12].flatten(), dtype=np.float32).reshape((-1, 3))
        v = np.frombuffer(data[:, 12:48].flatten(), dtype=np.float32).reshape((-1, 3, 3))
        v = v.reshape((-1, 3))
        f = np.arange(len(v)).reshape((-1, 3))
        return v, f, n

    def read_ascii_stl(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        v = []
        n = []
        for line in lines:
            if 'vertex' in line:
                v.append([float(x) for x in line.split()[1:4]])
            elif 'facet normal' in line:
                n.append([float(x) for x in line.split()[2:5]])
        v = np.array(v)
        n = np.array(n)
        f = np.arange(len(v)).reshape((-1, 3))
        return v, f, n

    # load the CT structure
    v, f, n = stlread('scaphoid.stl')
    # Calculate the centroid of the point cloud
    centroid = np.mean(v, axis=0)
    # Subtract the centroid from each point to center the point cloud
    centered_vertices = v - centroid
    original_CTptCloud = centered_vertices
    transformed_CTptCloud = original_CTptCloud

    # Load metadata (assuming it's the same for all frames)
    BfMeta = sio.loadmat(path + 'BfDataset_frame000.mat', simplify_cells=True).get('BfDataset').get('meta')

    # Initialize variables for template matching
    left_templates = [None] * N
    right_templates = [None] * N
    left_templateIndex = 0
    right_templateIndex = 0

    Transformation_Matrices = []
    cumulative_transformation = np.eye(4)

    for frame_idx in range(nFrames):
        print(f"Processing frame {frame_idx}")
        transformation_matrix, left_templates, right_templates, left_templateIndex, right_templateIndex, transformed_CTptCloud, smoothed_USptCloud = process_frame(
            frame_idx, BfMeta, transformed_CTptCloud, original_CTptCloud, left_templates, right_templates,
            left_templateIndex, right_templateIndex
        )
        Transformation_Matrices.append(transformation_matrix)

        # Update cumulative transformation
        cumulative_transformation = transformation_matrix @ cumulative_transformation

        # Apply cumulative transformation to the original CT point cloud
        CT_transformed = (cumulative_transformation[:3, :3] @ original_CTptCloud.T + cumulative_transformation[:3, 3][:,
                                                                                     np.newaxis]).T
        transformed_CTptCloud = CT_transformed

        # Visualization: Plot CT and US Point Clouds
        visualize_point_clouds(CT_transformed, smoothed_USptCloud, frame_idx)

    print("Processing complete. Transformation matrices:")
    for i, matrix in enumerate(Transformation_Matrices):
        print(f"Frame {i}:")
        print(matrix)
        print()