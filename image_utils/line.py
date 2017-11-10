import numpy as np


# Define a class to receive the characteristics of each line detection
class Line:
    # Define conversions in x and y from pixels space to meters
    Y_METERS_PER_PIXEL = 30 / 720  # meters per pixel in y dimension
    X_METERS_PER_PIXEL = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self, n_frames=5):
        # was the line detected in the last iteration?
        self.detected = False
        # the number of frames to consider
        self.nframes = n_frames
        # polynomials of the last n fits of the line
        self.last_n_polynomials = []
        # polynomial to use for plotting, etc
        self.best_poly = None
        # polynomial coefficients for the most recent fit
        self.current_poly = None  # array polynomial
        # polynomial coefficients for the previous fit
        self.previous_poly = None  # array polynomial
        # most recent batch of reliable x points and y points (parallel arrays)
        self.recent_plot_y = None
        self.recent_x_points = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # index to replace the oldest polynomial
        self.index_to_replace = 0

    def avg_poly(self):
        return np.sum(self.last_n_polynomials, axis=0) / len(self.last_n_polynomials)

    def add_or_replace_poly(self, polynomial):
        if len(self.last_n_polynomials) <= self.nframes:
            self.last_n_polynomials.append(polynomial)
        else:
            self.last_n_polynomials[self.index_to_replace] = polynomial
            self.index_to_replace = (self.index_to_replace + 1) % self.nframes

    def radius_of_curvature(self, y_eval):
        # Fit new polynomials to x,y in world space
        real_space_polynomial = np.polyfit(self.recent_plot_y * Line.Y_METERS_PER_PIXEL, self.recent_x_points * Line.X_METERS_PER_PIXEL, 2)
        # Calculate the new radii of curvature
        curve_radius = \
            ((1 + (2 * real_space_polynomial[0] * y_eval * Line.X_METERS_PER_PIXEL + real_space_polynomial[1]) ** 2) ** 1.5) / \
            np.absolute(2 * real_space_polynomial[0])
        return curve_radius


