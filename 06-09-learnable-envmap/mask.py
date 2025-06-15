"""
spherical environment visibility map generator
==============================================
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

def load_camera_data(file_path):
    """load camera data from json file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"error: could not find file {file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"error: invalid JSON in file {file_path}")
        sys.exit(1)

def extract_camera_info(data):
    """get camera positions, rotations, and viewing directions"""
    positions = []
    rotations = []
    view_directions = []
    
    for frame in data['frames']:
        transform = np.array(frame['transform_matrix'])
        position = transform[:3, 3]
        rotation = transform[:3, :3]
        view_direction = -rotation[:, 2]  # fwd direction
        
        positions.append(position)
        rotations.append(rotation)
        view_directions.append(view_direction)
    
    return np.array(positions), np.array(rotations), np.array(view_directions)

def cartesian_to_spherical(x, y, z):
    """conv cartesian coordinates to spherical (azimuth, elevation)"""
    # aximuth: angle around Z axis (0 to 2π)
    azimuth = np.arctan2(y, x)
    azimuth = (azimuth + 2 * np.pi) % (2 * np.pi)  # Ensure 0 to 2π
    
    # elevation: angle from xy plane (-π/2 to π/2)
    r = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arcsin(z / (r + 1e-10))  # + small value to avoid division by zero
    
    return azimuth, elevation

def spherical_to_equirectangular(azimuth, elevation, width, height):
    """Convert spherical coordinates to equirectangular image coordinates"""
    # norm to image coordinates
    u = azimuth / (2 * np.pi)  # 0 to 1
    v = (elevation + np.pi/2) / np.pi  # 0 to 1
    
    # conv to pixel coordinates
    x = u * width
    y = (1 - v) * height  # flip y axis for image coordinates
    
    return x, y

def compute_spherical_visibility(positions, rotations, fov_angle, 
                               width=1024, height=512, 
                               sample_rays_per_camera=1000):
    """
    Compute spherical environment visibility map
    
    Args:
        positions: Camera positions (N, 3)
        rotations: Camera rotation matrices (N, 3, 3)
        fov_angle: Field of view angle in radians
        width: Environment map width (typically 2:1 ratio)
        height: Environment map height
        sample_rays_per_camera: Number of rays to sample per camera
    
    Returns:
        visibility_map: 2D boolean array (height, width)
        spherical_info: Dictionary with parameters
    """
    
    print(f"Computing spherical visibility map ({width}x{height})...")
    print(f"Processing {len(positions)} cameras with {sample_rays_per_camera} rays each")
    print(f"Camera FOV: {np.degrees(fov_angle):.1f}°")
    
    # Initialize visibility map
    visibility = np.zeros((height, width), dtype=bool)
    
    # For each camera, sample rays within its field of view
    for cam_idx, (cam_pos, cam_rot) in enumerate(zip(positions, rotations)):
        if cam_idx % 10 == 0:
            print(f"Processing camera {cam_idx+1}/{len(positions)}")
        
        # Sample rays within camera's field of view
        sample_directions = sample_camera_fov(cam_rot, fov_angle, sample_rays_per_camera)
        
        # Convert directions to spherical coordinates
        for direction in sample_directions:
            # Direction is in world coordinates
            azimuth, elevation = cartesian_to_spherical(direction[0], direction[1], direction[2])
            
            # Convert to image coordinates
            img_x, img_y = spherical_to_equirectangular(azimuth, elevation, width, height)
            
            # Mark as visible (with bounds checking)
            x_idx = int(np.clip(img_x, 0, width - 1))
            y_idx = int(np.clip(img_y, 0, height - 1))
            visibility[y_idx, x_idx] = True
    
    # Calculate coverage statistics
    total_pixels = width * height
    visible_pixels = np.sum(visibility)
    coverage_percent = (visible_pixels / total_pixels) * 100
    
    spherical_info = {
        'width': width,
        'height': height,
        'coverage_percent': coverage_percent,
        'visible_pixels': int(visible_pixels),
        'total_pixels': total_pixels,
        'fov_degrees': float(np.degrees(fov_angle)),
        'projection': 'equirectangular'
    }
    
    print(f"Spherical visibility computation complete!")
    print(f"Coverage: {coverage_percent:.1f}% of environment sphere")
    
    return visibility, spherical_info

def sample_camera_fov(cam_rotation, fov_angle, num_samples):
    """
    Sample ray directions within a camera's field of view
    
    Args:
        cam_rotation: Camera rotation matrix (3x3)
        fov_angle: Field of view angle in radians
        num_samples: Number of ray directions to sample
    
    Returns:
        directions: Array of unit direction vectors in world coordinates (N, 3)
    """
    
    # Generate rays in camera space within FOV cone
    directions_cam = []
    
    # Sample uniformly within the FOV cone
    half_fov = fov_angle / 2
    
    for _ in range(num_samples):
        # Sample angles within FOV
        theta = np.random.uniform(-half_fov, half_fov)  # Horizontal angle
        phi = np.random.uniform(-half_fov, half_fov)    # Vertical angle
        
        # Convert to direction vector in camera space (looking down -Z)
        x_cam = np.tan(theta)
        y_cam = np.tan(phi)
        z_cam = -1.0  # Looking down negative Z
        
        # Normalize
        direction_cam = np.array([x_cam, y_cam, z_cam])
        direction_cam = direction_cam / np.linalg.norm(direction_cam)
        
        directions_cam.append(direction_cam)
    
    directions_cam = np.array(directions_cam)
    
    # Transform to world coordinates
    directions_world = (cam_rotation @ directions_cam.T).T
    
    return directions_world

def create_environment_map_visualization(visibility, spherical_info, 
                                       gray_color=128, visible_color=255):
    """
    Create environment map visualization like the examples shown
    
    Args:
        visibility: Boolean visibility array
        spherical_info: Info dictionary
        gray_color: Color value for non-visible areas (0-255)
        visible_color: Color value for visible areas (0-255)
    
    Returns:
        env_map: RGB environment map array
    """
    
    height, width = visibility.shape
    
    # Create RGB environment map
    env_map = np.full((height, width, 3), gray_color, dtype=np.uint8)
    
    # For visible areas, create some texture/pattern
    # You can replace this with actual environment texture if available
    visible_mask = visibility
    
    # Create a simple pattern for visible areas
    for y in range(height):
        for x in range(width):
            if visible_mask[y, x]:
                # Create a simple checkered or noise pattern
                noise = np.random.randint(0, 255, 3)
                env_map[y, x] = noise
    
    return env_map

def save_spherical_results(visibility, spherical_info, output_prefix='spherical_visibility'):
    """Save spherical visibility results"""
    
    # Save numpy array
    np.save(f'results/{output_prefix}_map.npy', visibility)
    
    # Save info
    with open(f'results/{output_prefix}_info.json', 'w') as f:
        json.dump(spherical_info, f, indent=2)
    
    # Create and save environment map visualization
    env_map = create_environment_map_visualization(visibility, spherical_info)
    
    # Save as image
    from PIL import Image
    img = Image.fromarray(env_map)
    img.save(f'results/{output_prefix}_environment.png')
    
    # Create matplotlib visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Binary visibility
    axes[0,0].imshow(visibility, cmap='binary', aspect='auto')
    axes[0,0].set_title('Binary Visibility (Black=Hidden, White=Visible)')
    axes[0,0].set_xlabel('Azimuth (longitude)')
    axes[0,0].set_ylabel('Elevation (latitude)')
    
    # Gray-scale environment map
    gray_env = create_environment_map_visualization(visibility, spherical_info, 
                                                   gray_color=128, visible_color=255)
    axes[0,1].imshow(gray_env, aspect='auto')
    axes[0,1].set_title('Environment Map (Gray=Hidden)')
    axes[0,1].set_xlabel('Azimuth (longitude)')
    axes[0,1].set_ylabel('Elevation (latitude)')
    
    # Coverage statistics
    height, width = visibility.shape
    
    # Coverage by elevation bands
    elevation_bands = np.linspace(-90, 90, 10)
    band_coverage = []
    for i in range(len(elevation_bands) - 1):
        start_y = int((1 - (elevation_bands[i+1] + 90) / 180) * height)
        end_y = int((1 - (elevation_bands[i] + 90) / 180) * height)
        band_vis = visibility[start_y:end_y, :]
        coverage = np.mean(band_vis) * 100 if band_vis.size > 0 else 0
        band_coverage.append(coverage)
    
    axes[1,0].bar(range(len(band_coverage)), band_coverage, 
                  color='skyblue', alpha=0.7)
    axes[1,0].set_title('Coverage by Elevation Bands')
    axes[1,0].set_xlabel('Elevation Band (Bottom to Top)')
    axes[1,0].set_ylabel('Coverage %')
    
    # Coverage by azimuth
    azimuth_coverage = np.mean(visibility, axis=0) * 100
    azimuth_angles = np.linspace(0, 360, width)
    axes[1,1].plot(azimuth_angles, azimuth_coverage, 'b-', linewidth=2)
    axes[1,1].set_title('Coverage by Azimuth (360° view)')
    axes[1,1].set_xlabel('Azimuth (degrees)')
    axes[1,1].set_ylabel('Coverage %')
    axes[1,1].set_xlim(0, 360)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Add grid lines for reference
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(env_map, aspect='auto', extent=[0, 360, -90, 90])
    
    # Add longitude lines every 30 degrees
    for lon in range(0, 361, 30):
        ax.axvline(lon, color='white', alpha=0.3, linewidth=0.5)
    
    # Add latitude lines every 30 degrees  
    for lat in range(-90, 91, 30):
        ax.axhline(lat, color='white', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title(f'Spherical Environment Visibility Map\n{spherical_info["coverage_percent"]:.1f}% Coverage')
    plt.savefig(f'results/{output_prefix}_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Spherical results saved:")
    print(f"  - {output_prefix}_map.npy (binary visibility array)")
    print(f"  - {output_prefix}_environment.png (environment map)")
    print(f"  - {output_prefix}_analysis.png (detailed analysis)")
    print(f"  - {output_prefix}_grid.png (with coordinate grid)")
    print(f"  - {output_prefix}_info.json (parameters)")

def create_spherical_visibility_map(json_file_path, output_prefix='spherical_env',
                                  width=1024, height=512, rays_per_camera=2000):
    """
    Main function to create spherical environment visibility maps
    
    Args:
        json_file_path: Path to camera JSON file
        output_prefix: Output file prefix
        width: Environment map width (longitude resolution)
        height: Environment map height (latitude resolution)
        rays_per_camera: Number of rays to sample per camera
    """
    
    print(f"Creating spherical environment visibility map...")
    print(f"Input: {json_file_path}")
    print(f"Output resolution: {width}x{height}")
    
    # Load camera data
    data = load_camera_data(json_file_path)
    positions, rotations, view_directions = extract_camera_info(data)
    fov_angle = data['camera_angle_x']
    
    print(f"Loaded {len(positions)} cameras")
    print(f"Camera FOV: {np.degrees(fov_angle):.1f}°")
    
    # Compute spherical visibility
    visibility, info = compute_spherical_visibility(
        positions, rotations, fov_angle,
        width=width, height=height,
        sample_rays_per_camera=rays_per_camera
    )
    
    # Save results
    save_spherical_results(visibility, info, output_prefix)
    
    return visibility, info

def main():
    """Main function with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate spherical environment visibility maps')
    parser.add_argument('input_file', help='Path to camera JSON file')
    parser.add_argument('--output', '-o', default='spherical_env', 
                       help='Output prefix (default: spherical_env)')
    parser.add_argument('--width', '-w', type=int, default=1024,
                       help='Environment map width (default: 1024)')
    parser.add_argument('--height', '-H', type=int, default=512,
                       help='Environment map height (default: 512)')
    parser.add_argument('--rays', '-r', type=int, default=2000,
                       help='Rays per camera (default: 2000)')
    
    args = parser.parse_args()
    
    visibility, info = create_spherical_visibility_map(
        args.input_file,
        output_prefix=args.output,
        width=args.width,
        height=args.height,
        rays_per_camera=args.rays
    )
    
    print(f"\nSphericl Environment Map Complete!")
    print(f"Coverage: {info['coverage_percent']:.1f}% of full sphere")
    print(f"Resolution: {info['width']}x{info['height']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Default run
        json_file = 'data/body_10_30/transforms_train.json'
        
        if os.path.exists(json_file):
            # Create high-resolution spherical environment map
            visibility, info = create_spherical_visibility_map(
                json_file,
                output_prefix='body_spherical_env',
                width=2048,  # High resolution for detailed environment map
                height=1024,
                rays_per_camera=3000  # More rays for better coverage
            )
        else:
            print(f"Usage: python {sys.argv[0]} <camera_json_file>")
            print(f"Example: python {sys.argv[0]} data/body_10_30/transforms_train.json")