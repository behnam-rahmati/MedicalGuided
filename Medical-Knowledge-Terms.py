###################################################################################  Largest connected Mask Term############################################################################################################
import numpy as np
import cv2

# Assuming pred_masks_all contains all the masks and R is the reliability mask
# Modify the shape of R to match the masks if necessary

largest_reliable_mask = None
max_reliable_pixels = -1

for mask in pred_masks_all[slice_index]:
    # Multiply the mask with the reliability mask
    mask_with_reliability = mask * R

    # Extract contours from the modified mask
    tmp = mask_with_reliability.astype(np.uint8)
    coords, _ = cv2.findContours(tmp.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if not coords:
        # No contours found, continue to the next mask
        continue

    # Find the largest contour based on the number of reliable pixels
    largest_contour = max(coords, key=lambda x: cv2.contourArea(x))

    # Calculate the number of reliable pixels in the largest contour
    num_reliable_pixels = np.sum(mask_with_reliability)

    # Update the largest reliable mask if this contour has more reliable pixels
    if num_reliable_pixels > max_reliable_pixels:
        max_reliable_pixels = num_reliable_pixels
        largest_reliable_mask = mask

# Now largest_reliable_mask contains the mask with the highest number of reliable pixels


######################################################################################Filling the gaps/ holes (first solution)#######################################################################################################


# Read the binary mask
mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to draw filled contours
filled_mask = np.zeros_like(mask)

# Fill the contours
cv2.fillPoly(filled_mask, contours, color=255)

# Save the filled mask
cv2.imwrite('filled_mask_contours.png', filled_mask)
######################################################################################Filling the gaps/ holes (second solution)#######################################################################################################


# Read the binary mask
mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

# Apply morphological closing to fill the holes
kernel = np.ones((5, 5), np.uint8)
filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Save the filled mask
cv2.imwrite('filled_mask_morphology.png', filled_mask)


################################################################################### DEFORMABLE MODELS TERM############################################################################################################
### 1- deformable models
# *********This is a modified version of the original code of Shawn Lankton***********
# shape prior term is added to the original code

# chan-vese with shape prior

# -------------------------------------------------------------------------------------
# Region Based Active Contour Segmentation
#
#
# Inputs: I           2D image
#         init_mask   Initialization (1 = foreground, 0 = bg)
#         max_its     Number of iterations to run segmentation for
#         alpha       (optional) Weight of smoothing term
#                       higer = smoother.  default = 0.2
#         display     (optional) displays intermediate outputs
#                       default = true
#        shape_w      weight of shape prior term           
#
# Outputs: seg        Final segmentation mask (1=fg, 0=bg)
#
# Description: This code implements the paper: "Active Contours Without
# Edges" By Chan Vese. This is a nice way to segment images whose
# foregrounds and backgrounds are statistically different and homogeneous.
#
# Coded by: Shawn Lankton (www.shawnlankton.com)
# ------------------------------------------------------------------------
# 
# Gif montage (save each step in a gif folder):
# convert -delay 50 gif/levelset*.png levelset.gif

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


eps = np.finfo(float).eps


def chanvese( I, init_mask ,shape_w ,max_its=200, alpha=0.2,
             thresh=0, color='r', display=False ):
    I = I.astype(np.float)

    # Create a signed distance map (SDF) from mask
    phi = mask2phi(init_mask)

    if display:
        plt.ion()
        fig, axes = plt.subplots(ncols=2)
        show_curve_and_phi(fig, I, phi, color)
        plt.savefig('levelset_start.png', bbox_inches='tight')

    # Main loop
    its = 0
    stop = False
    prev_mask = init_mask
    c = 0
    phi_LV = phi
    while (its < max_its and not stop):
        # Get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) > 0:
            # Intermediate output
            if display:
                if np.mod(its, 50) == 0:
                    print('iteration: {0}'.format(its))
                    show_curve_and_phi(fig, I, phi, color)
            else:
                if np.mod(its, 10) == 0:
                    pass
                    # print('iteration: {0}'.format(its))

            # Find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(I.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(I.flat[vpts]) / (len(vpts) + eps)  # exterior mean

            # Force from image information
            F = (I.flat[idx] - u)**2 - (I.flat[idx] - v)**2
            # Force from curvature penalty
            curvature = get_curvature(phi, idx)
            
            dEdl= -(phi.flat[idx]-phi_LV.flat[idx]);

            # Gradient descent to minimize energy
            dphidt = F / np.max(np.abs(F)) + alpha * curvature +shape_w * dEdl

            # Maintain the CFL condition
            dt = 0.45 / (np.max(np.abs(dphidt)) + eps)

            # Evolve the curve
            phi.flat[idx] += dt * dphidt

            # Keep SDF smooth
            phi = sussman(phi, 0.5)

            # new_mask = phi <= 0
            # c = convergence(prev_mask, new_mask, thresh, c)

            # if c <= 5:
            its = its + 1
                # prev_mask = new_mask
            # else:
                # stop = True

        else:
            break

    # Final output
    if display:
        show_curve_and_phi(fig, I, phi, color)
        plt.savefig('levelset_end.png', bbox_inches='tight')

    # Make mask from SDF
    seg = phi <= 0  # Get mask from levelset

    return seg, phi, its


# ---------------------------------------------------------------------
# ---------------------- AUXILIARY FUNCTIONS --------------------------
# ---------------------------------------------------------------------

def bwdist(a):
    """
    Intermediary function. 'a' has only True/False vals,
    so we convert them into 0/1 values - in reverse.
    True is 0, False is 1, distance_transform_edt wants it that way.
    """
    return nd.distance_transform_edt(a == 0)


# Displays the image with curve superimposed
def show_curve_and_phi(fig, I, phi, color):
    fig.axes[0].cla()
    fig.axes[0].imshow(I, cmap='gray')
    fig.axes[0].contour(phi, 0, colors=color)
    fig.axes[0].set_axis_off()
    plt.draw()

    fig.axes[1].cla()
    fig.axes[1].imshow(phi)
    fig.axes[1].set_axis_off()
    plt.draw()
    
    plt.pause(0.1)


def im2double(a):
    a = a.astype(np.float)
    a /= np.abs(a).max()
    return a


# Converts a mask to a SDF
def mask2phi(init_a):
    phi = bwdist(init_a) - bwdist(1 - init_a) + im2double(init_a) - 0.5
    return phi


# Compute curvature along SDF
def get_curvature(phi, idx):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                     phi.flat[iddr] + phi.flat[idul])
    phi_x2 = phi_x**2
    phi_y2 = phi_y**2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + eps) ** 1.5) * (phi_x2 + phi_y2) ** 0.5

    return curvature


# Level set re-initialization by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - np.roll(D, 1, axis=1)
    b = np.roll(D, -1, axis=1) - D
    c = D - np.roll(D, -1, axis=0)
    d = np.roll(D, 1, axis=0) - D

    a_p = np.clip(a, 0, np.inf)
    a_n = np.clip(a, -np.inf, 0)
    b_p = np.clip(b, 0, np.inf)
    b_n = np.clip(b, -np.inf, 0)
    c_p = np.clip(c, 0, np.inf)
    c_n = np.clip(c, -np.inf, 0)
    d_p = np.clip(d, 0, np.inf)
    d_n = np.clip(d, -np.inf, 0)

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros_like(D)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_p.flat[D_pos_ind]**2], [b_n.flat[D_pos_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_p.flat[D_pos_ind]**2], [d_n.flat[D_pos_ind]**2])), axis=0)) - 1
    dD.flat[D_neg_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_n.flat[D_neg_ind]**2], [b_p.flat[D_neg_ind]**2])), axis=0) +
        np.max(np.concatenate(
            ([c_n.flat[D_neg_ind]**2], [d_p.flat[D_neg_ind]**2])), axis=0)) - 1

    D = D - dt * sussman_sign(D) * dD
    return D


def sussman_sign(D):
    return D / np.sqrt(D**2 + 1)


# Convergence Test
def convergence(p_mask, n_mask, thresh, c):
    diff = p_mask - n_mask
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
    return c


if __name__ == "__main__":
    img = nd.imread('brain.png', flatten=True)
    mask = np.zeros(img.shape)
    mask[20:100, 20:100] = 1

    chanvese(img, mask, max_its=1000, display=True, alpha=1.0)


################################################################################### Total variation TERM############################################################################################################
import tensorflow as tf

def custom_loss(y_true, y_pred):
    # Define your loss function (e.g., cross-entropy for segmentation)
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    # Spatial regularization parameter (lambda)
    lambda_reg = 0.01
    
    # Calculate spatial regularization term (e.g., total variation regularization)
    diff_i = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    diff_j = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    spatial_reg = lambda_reg * (tf.reduce_sum(diff_i) + tf.reduce_sum(diff_j))
    
    # Total loss with regularization
    total_loss = cross_entropy_loss + spatial_reg
    
    return total_loss


################################################################################### Convex Hull ############################################################################################################

# Read the binary mask
mask = cv2.imread('binary_mask.png', cv2.IMREAD_GRAYSCALE)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming there's only one contour, you can take the first one
contour = contours[0]

# Find the convex hull of the contour
convex_hull = cv2.convexHull(contour)

# Create a blank image to draw the convex hull
convex_hull_img = np.zeros_like(mask)

# Draw the convex hull on the blank image
cv2.drawContours(convex_hull_img, [convex_hull], 0, 255, -1)

# Save the convex hull image
cv2.imwrite('convex_hull.png', convex_hull_img)


