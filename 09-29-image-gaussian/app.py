import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
from PIL import Image

plt.style.use('default')

@dataclass
class Gaussian2D:
    mu_x: float
    mu_y: float
    sigma_xx: float
    sigma_yy: float
    sigma_xy: float
    weight: float
    mean_color: np.ndarray
    sigma_xc: np.ndarray
    sigma_cc: np.ndarray
    
    def __post_init__(self):
        if isinstance(self.mean_color, (list, tuple)):
            self.mean_color = np.array(self.mean_color)
        if isinstance(self.sigma_xc, (list, tuple)):
            self.sigma_xc = np.array(self.sigma_xc)
        if isinstance(self.sigma_cc, (list, tuple)):
            self.sigma_cc = np.array(self.sigma_cc)
    
    @property
    def spatial_covariance(self) -> np.ndarray:
        return np.array([[self.sigma_xx, self.sigma_xy],
                        [self.sigma_xy, self.sigma_yy]])
    
    @property 
    def spatial_mean(self) -> np.ndarray:
        return np.array([self.mu_x, self.mu_y])
    
    def responsibility_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pos = np.dstack((x, y))
        return self.weight * multivariate_normal.pdf(
            pos, self.spatial_mean, self.spatial_covariance
        )
    
    def conditional_color_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pos_diff = np.stack([x - self.mu_x, y - self.mu_y], axis=-1)
        
        try:
            sigma_xx_inv = np.linalg.inv(self.spatial_covariance)
            color_offset = np.einsum('...i,ij,jk->...k', pos_diff, sigma_xx_inv, self.sigma_xc)
            conditional_color = self.mean_color + color_offset
        except:
            conditional_color = np.broadcast_to(self.mean_color, pos_diff.shape[:-1] + (len(self.mean_color),))
        
        return conditional_color
    
    def get_bounding_box(self, n_sigma: float = 3.0) -> Tuple[int, int, int, int]:
        eigenvals = np.linalg.eigvals(self.spatial_covariance)
        max_std = np.sqrt(np.max(eigenvals))
        
        radius = n_sigma * max_std
        x_min = int(np.floor(self.mu_x - radius))
        x_max = int(np.ceil(self.mu_x + radius))
        y_min = int(np.floor(self.mu_y - radius))
        y_max = int(np.ceil(self.mu_y + radius))
        
        return x_min, y_min, x_max, y_max

class GaussianImageRepresentation:
    def __init__(self, width: int, height: int, channels: int = 3):
        self.width = width
        self.height = height
        self.channels = channels
        self.gaussians: List[Gaussian2D] = []
    
    def add_gaussian(self, gaussian: Gaussian2D):
        self.gaussians.append(gaussian)
    
    def render(self, supersample: int = 1) -> np.ndarray:
        render_width = self.width * supersample
        render_height = self.height * supersample
        
        x = np.linspace(0, self.width - 1, render_width)
        y = np.linspace(0, self.height - 1, render_height)
        X, Y = np.meshgrid(x, y)
        
        if self.channels == 1:
            image = np.zeros((render_height, render_width))
            total_responsibility = np.zeros((render_height, render_width))
        else:
            image = np.zeros((render_height, render_width, self.channels))
            total_responsibility = np.zeros((render_height, render_width))
        
        for gaussian in self.gaussians:
            x_min, y_min, x_max, y_max = gaussian.get_bounding_box()
            
            x_min = max(0, int(x_min * supersample))
            y_min = max(0, int(y_min * supersample))
            x_max = min(render_width, int(x_max * supersample))
            y_max = min(render_height, int(y_max * supersample))
            
            if x_min >= x_max or y_min >= y_max:
                continue
                
            X_region = X[y_min:y_max, x_min:x_max]
            Y_region = Y[y_min:y_max, x_min:x_max]
            
            if X_region.size == 0 or Y_region.size == 0:
                continue
            
            responsibility = gaussian.responsibility_at(X_region, Y_region)
            conditional_color = gaussian.conditional_color_at(X_region, Y_region)
            
            region_shape = (y_max - y_min, x_max - x_min)
            responsibility = responsibility.reshape(region_shape)
            
            total_responsibility[y_min:y_max, x_min:x_max] += responsibility
            
            if self.channels == 1:
                conditional_color = conditional_color.reshape(region_shape)
                image[y_min:y_max, x_min:x_max] += responsibility * conditional_color
            else:
                conditional_color = conditional_color.reshape(region_shape + (self.channels,))
                weighted_color = responsibility[:, :, np.newaxis] * conditional_color
                image[y_min:y_max, x_min:x_max, :] += weighted_color
        
        mask = total_responsibility > 1e-8
        if self.channels == 1:
            image[mask] /= total_responsibility[mask]
        else:
            image[mask, :] /= total_responsibility[mask, np.newaxis]
        
        if supersample > 1:
            if self.channels == 1:
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
            else:
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def get_compression_ratio(self, original_image: np.ndarray) -> float:
        original_size = original_image.size * 8
        gaussian_size = (6 + self.channels) * 32
        compressed_size = len(self.gaussians) * gaussian_size
        return original_size / compressed_size
    
    def save_gaussians(self, filename: str):
        data = []
        for g in self.gaussians:
            data.append([g.mu_x, g.mu_y, g.sigma_xx, g.sigma_yy, g.sigma_xy, 
                        g.amplitude] + g.color.tolist())
        np.save(filename, data)
    
    def load_gaussians(self, filename: str):
        data = np.load(filename)
        self.gaussians = []
        for row in data:
            color = row[6:6+self.channels]
            gaussian = Gaussian2D(
                mu_x=row[0], mu_y=row[1],
                sigma_xx=row[2], sigma_yy=row[3], sigma_xy=row[4],
                weight=row[5], mean_color=color,
                sigma_xc=np.zeros((2, len(color))), sigma_cc=np.eye(len(color))
            )
            self.gaussians.append(gaussian)

class GaussianFitter:
    @staticmethod
    def fit_mixture_model(image: np.ndarray, n_components: int = 50) -> GaussianImageRepresentation:
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
            colors = image.flatten()
        else:
            height, width, channels = image.shape
            colors = image.reshape(-1, channels)
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        positions = np.column_stack([x_coords.flatten(), y_coords.flatten()])
        
        pos_norm = positions / np.array([width, height])
        color_norm = colors / 255.0
        
        features = np.column_stack([pos_norm, color_norm])
        
        gmm = GaussianMixture(n_components=n_components, random_state=42, 
                             covariance_type='full', max_iter=1000)
        gmm.fit(features)
        
        gaussian_repr = GaussianImageRepresentation(width, height, channels)
        
        for i in range(n_components):
            full_cov = gmm.covariances_[i]
            full_mean = gmm.means_[i]
            
            spatial_cov = full_cov[:2, :2]
            spatial_cov[0, 0] *= width**2
            spatial_cov[1, 1] *= height**2
            spatial_cov[0, 1] *= width * height
            spatial_cov[1, 0] *= width * height
            
            color_cov = full_cov[2:, 2:]
            cross_cov = full_cov[2:, :2]
            cross_cov[:, 0] *= width
            cross_cov[:, 1] *= height
            
            mu_x = full_mean[0] * width
            mu_y = full_mean[1] * height
            mean_color = full_mean[2:] * 255.0
            
            eigenvals = np.linalg.eigvals(spatial_cov)
            if np.any(eigenvals <= 0):
                spatial_cov = np.eye(2) * (min(width, height) / 10)**2
                cross_cov = np.zeros_like(cross_cov)
            
            gaussian = Gaussian2D(
                mu_x=mu_x, mu_y=mu_y,
                sigma_xx=spatial_cov[0, 0],
                sigma_yy=spatial_cov[1, 1], 
                sigma_xy=spatial_cov[0, 1],
                weight=gmm.weights_[i],
                mean_color=mean_color if channels > 1 else np.array([mean_color[0]]),
                sigma_xc=cross_cov.T,
                sigma_cc=color_cov
            )
            gaussian_repr.add_gaussian(gaussian)
        
        return gaussian_repr
    

def visualize_gaussians(gaussian_repr: GaussianImageRepresentation, 
                        original_image: Optional[np.ndarray] = None,
                        show_individual: bool = False):
    rendered = gaussian_repr.render()
    
    if original_image is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        if len(original_image.shape) == 2:
            axes[0].imshow(original_image, cmap='gray')
        else:
            axes[0].imshow(original_image)
        axes[0].set_title('original')
        axes[0].axis('off')
        
        if len(rendered.shape) == 2:
            axes[1].imshow(rendered, cmap='gray')
        else:
            axes[1].imshow(rendered)
        axes[1].set_title(f'gaussians ({len(gaussian_repr.gaussians)})')
        axes[1].axis('off')
        
        if len(original_image.shape) == 2:
            diff = np.abs(original_image.astype(float) - rendered.astype(float))
            im = axes[2].imshow(diff, cmap='hot')
            plt.colorbar(im, ax=axes[2])
        else:
            diff = np.mean(np.abs(original_image.astype(float) - rendered.astype(float)), axis=2)
            im = axes[2].imshow(diff, cmap='hot')
            plt.colorbar(im, ax=axes[2])
            
        axes[2].set_title(f'error: {np.mean(diff):.1f}')
        axes[2].axis('off')
        
        compression_ratio = gaussian_repr.get_compression_ratio(original_image)
        print(f"compression: {compression_ratio:.1f}x")
        
    else:
        plt.figure(figsize=(8, 6))
        if len(rendered.shape) == 2:
            plt.imshow(rendered, cmap='gray')
        else:
            plt.imshow(rendered)
        plt.title(f'gaussians ({len(gaussian_repr.gaussians)})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

haystack_image = np.array(Image.open("heystacks.jpg"))
haystack_resized = cv2.resize(haystack_image, (160, 120))

print("fitting gaussians to img")
gmm_repr = GaussianFitter.fit_mixture_model(haystack_resized, n_components=1000)
visualize_gaussians(gmm_repr, haystack_resized)