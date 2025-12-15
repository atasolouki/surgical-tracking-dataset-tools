function [Out ]= scanConversion(scan_lines, ~, depths, resolution)
%SCANCONVERSION Convert scan-lines in polar coordinates to a B-mode ultrasound image.
%
% DESCRIPTION:
%       scanConversion computes the remapping between a series of scan
%       lines in polar coordinates (i.e., taken at different steering
%       angles) to a B-mode ultrasound image in Cartesian coordinates
%       suitable for display. The remapping is performed using bilinear
%       interpolation via interp2. 
%
% USAGE:
%       b_mode = scanConversion(scan_lines, steering_angles, image_size, c0, dt)
%       b_mode = scanConversion(scan_lines, steering_angles, image_size, c0, dt, resolution)
%
% INPUTS:
%       scan_lines      - matrix of scan lines indexed as (time, angle)
%       steering_angles - array of scanning angles [degrees]
%       depths          - array of sample depths [metres]
%       resolution      - optional input to set the resolution of the
%                         output images in pixels (default = [256, 256])
%
% OUTPUTS:
%       b_mode          - the converted B-mode ultrasound image
%
% ABOUT:
%       author      - Bradley E. Treeby
%       date        - 23rd February 2011
%       last update - 12th September 2012
%
% This function is part of the k-Wave Toolbox (http://www.k-wave.org)
% Copyright (C) 2009-2014 Bradley Treeby and Ben Cox
%
% See also interp2

% This file is part of k-Wave. k-Wave is free software: you can
% redistribute it and/or modify it under the terms of the GNU Lesser
% General Public License as published by the Free Software Foundation,
% either version 3 of the License, or (at your option) any later version.
% 
% k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
% more details. 
% 
% You should have received a copy of the GNU Lesser General Public License
% along with k-Wave. If not, see <http://www.gnu.org/licenses/>.

% define literals
X_RESOLUTION_DEF = 4096;     % [pixels]
Y_RESOLUTION_DEF = 4096;     % [pixels]

% check for the resolution inputs
if nargin == 4
    x_resolution = resolution(1);
    y_resolution = resolution(end);
else
    x_resolution = X_RESOLUTION_DEF;
    y_resolution = Y_RESOLUTION_DEF;
end


% Get image boundaries
dep_min = depths(1);
dep_max = depths(end);
lat_min = -19.2e-3;
lat_max =  19.2e-3;

% create regular Cartesian grid to remap to
pos_vec_y_new = (0:1/(x_resolution-1):1).*(lat_max-lat_min) + lat_min;
pos_vec_x_new = (0:1/(y_resolution-1):1).*(dep_max-dep_min) + dep_min;

[pos_mat_y_new, pos_mat_x_new] = ndgrid(pos_vec_y_new, pos_vec_x_new);

% convert new points to polar coordinates
apex_positions = linspace(lat_min,lat_max,98);

% interpolate using linear interpolation
b_mode = interp2(apex_positions,depths,scan_lines, pos_mat_y_new , pos_mat_x_new, 'linear').';

Bmode_mask = ones(size(b_mode));
Bmode_mask(isnan(b_mode)) = 0;
b_mode(isnan(b_mode)) = min(min(b_mode));
b_mode(isinf(b_mode)) = min(b_mode(~isinf(b_mode)));

Out.B_mode = b_mode;
Out.Bmode_mask = Bmode_mask;
Out.disp_az_lims = pos_vec_y_new;
Out.disp_dep_lims = pos_vec_x_new;

end