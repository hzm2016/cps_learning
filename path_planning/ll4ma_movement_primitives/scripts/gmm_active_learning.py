import os
import numpy as np
import random
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, animation
from tqdm import tqdm


class GMMActiveLearning:

    def __init__(self, foldername, num_frames=5, grid_dim=20, neg_starts=[], neg_extents=[],
                 dp_prior=True, random_sample=False, neg_border_width=0):
        self.folder_path = os.path.expanduser("~/Videos/gmm_active_learning/%s" % foldername)
        os.makedirs(self.folder_path)
        self.num_frames = num_frames
        self.grid_dim = grid_dim
        self.random_sample = random_sample
        if dp_prior:
            self.pos_gmm = DPGMM()
            self.neg_gmm = DPGMM()
        else:
            self.pos_gmm = GMM()
            self.neg_gmm = GMM()

        self.num_samples = 0
        self.num_neg = 0
        self.num_pos = 0
        self.tot_num_neg = 0
        self.candidates = self.get_data(neg_starts, neg_extents, neg_border_width)
        self.full_data = np.copy(self.candidates)
        self.selected_scatter = None # Visualization of selected sample point
        self.tot_entropies = []
        self.scatter = None
        self.first_iter = True
        self.progress_bar = None
        self.img_idx = 0
                
    def run(self):
        print ("\n\n===========================================================================\n"
               "  Running GMM active learning:\n"
               "    Folder: %s\n"
               "    Num Iterations: %d\n"
               "    Random Sample: %r\n"
               "===========================================================================\n\n"
               % (self.folder_path, self.num_frames, self.random_sample))
               
        # Get a few samples of each type to initiate GMM learning
        pos_data = self.candidates[np.where(self.candidates[:,0] == 1)][:,1:]
        neg_data = self.candidates[np.where(self.candidates[:,0] == 0)][:,1:]
        self.pos_samples = pos_data[np.random.randint(pos_data.shape[0], size=4),:]
        self.neg_samples = neg_data[np.random.randint(neg_data.shape[0], size=4),:]

        self.num_pos = self.pos_samples.shape[0]
        self.num_neg = self.neg_samples.shape[0]
        self.num_samples = self.num_pos + self.num_neg
        
        # Perform initial fit of the GMMs
        self.pos_gmm.fit(self.pos_samples)
        self.neg_gmm.fit(self.neg_samples)

        for i in range(self.num_frames):
            self.animate(i)
        
        # ani = animation.FuncAnimation(self.fig, self.animate, frames=self.num_frames)
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=5, bitrate=1800)
        # ani.save('video.mp4', writer=writer)
        print "\n\nComplete.\n"
        
    def animate(self, itr_num):
        # Have to do this as it seems the first iteration doesn't provide any functionality
        if self.first_iter:
            self.progress_bar = tqdm(total=self.num_frames)
            self.first_iter = False
        else:
            self.progress_bar.update(1)
            
        # # Clear drawn GMMs
        # self.axes[0].artists = []
        # self.axes[0].set_ylim(-1,self.grid_dim)
        # self.axes[0].set_xlim(-1,self.grid_dim)
        
        # if self.scatter is not None:
        #     self.scatter.remove()
            
        # Visualize GMMs before update since probs are computed based on these ones
        # self.plot_gmm(self.pos_samples, self.pos_gmm, self.axes[0], "cornflowerblue")
        # self.plot_gmm(self.neg_samples, self.neg_gmm, self.axes[0], "firebrick")
    
        # Compute probs/measures over all data for visualization
        full_pos_probs, full_neg_probs = self.get_probs(self.full_data)
        full_entropies = self.get_entropies(full_pos_probs, full_neg_probs)

        # Get the next sample
        pos_probs, neg_probs = self.get_probs(self.candidates)
        entropies = self.get_entropies(pos_probs, neg_probs)
        if self.random_sample:
            chosen_idx  = np.random.randint(0, len(entropies))
        else:
            # Find idx of max value, collect set of all max vals (ties), pick a random one of those
            max_entropy = 0.0
            for point, entropy in entropies:
                max_entropy = max(max_entropy, entropy)
            max_idxs = [idx for idx, value in enumerate(entropies) if value[1] == max_entropy]
            chosen_idx = np.random.choice(max_idxs)
        next_sample, _ = entropies[chosen_idx]
        if next_sample[0] == 1:
            self.pos_samples = np.vstack((self.pos_samples, next_sample[1:]))
            self.num_pos += 1
        else:
            self.neg_samples = np.vstack((self.neg_samples, next_sample[1:]))
            self.num_neg += 1
        self.candidates = np.delete(self.candidates, chosen_idx, axis=0)
        self.num_samples += 1

        # Compute total map entropy
        tot_entropy = 0.0
        for point, entropy in full_entropies:
            tot_entropy += entropy
        self.tot_entropies.append(tot_entropy)

        # Re-fit the GMMS
        self.pos_gmm.n_components = min(10, self.pos_samples.shape[0])
        self.neg_gmm.n_components = min(10, self.neg_samples.shape[0])
        self.pos_gmm.fit(self.pos_samples)
        self.neg_gmm.fit(self.neg_samples)

        self.plot(next_sample, full_entropies)

            
    def plot(self, next_sample, entropies):
        fig, axes = plt.subplots(1,3)
        axes = axes.flatten()
        fig.set_size_inches(15,5)
        for ax in axes:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # Visualize scatter plot
        axes[0].set_title("Pos: %d/%d, Neg: %d/%d, Tot: %d/%d"
                               % (self.num_pos, self.grid_dim**2 - self.tot_num_neg,
                                  self.num_neg, self.tot_num_neg,
                                  self.num_samples, self.grid_dim**2), fontsize=14)
        axes[0].set_ylim(-1,self.grid_dim)
        axes[0].set_xlim(-1,self.grid_dim)
        self.plot_scatter_data(self.pos_samples, axes[0], "cornflowerblue")
        self.plot_scatter_data(self.neg_samples, axes[0], "firebrick")
        self.plot_scatter_data(next_sample, axes[0], "gold", large=True)
        self.plot_gmm(self.pos_samples, self.pos_gmm, axes[0], "cornflowerblue")
        self.plot_gmm(self.neg_samples, self.neg_gmm, axes[0], "firebrick")

    
        # Visualize current map entropy
        axes[1].set_title("Entropy", fontsize=18)
        self.plot_grid_img(entropies, axes[1])
        # self.plot_grid_img(full_pos_probs, self.axes[1])
    
        # Visualize total map entropy over learning iterations
        axes[2].set_title("Total Map Entropy", fontsize=18)
        axes[2].set_ylim(0, 0.7 * self.grid_dim**2)
        axes[2].set_xlim(0, self.num_frames)
        self.plot_line(self.tot_entropies, axes[2])
        plt.tight_layout()
        plt.savefig("%s/img_%04d.png" % (self.folder_path, self.img_idx))
        self.img_idx += 1
        plt.close(fig)

    def get_data(self, neg_starts=[], neg_extents=[], neg_border_width=0):
        """
        neg_starts: list of bottom-left corner of negative regions
        neg_ends:   list of side extents for negative regions (all are square)
        """
        neg_pts = []
        if neg_border_width > 0:
            neg_starts = [(s1 + neg_border_width, s2 + neg_border_width) for s1, s2 in neg_starts]
            for x in range(self.grid_dim + 2*neg_border_width):
                for y in range(self.grid_dim + 2*neg_border_width):
                    if (x < neg_border_width or x > self.grid_dim + neg_border_width-1 or
                        y < neg_border_width or y > self.grid_dim + neg_border_width-1):
                        neg_pts.append((x,y))
        for start, extent in zip(neg_starts, neg_extents):
            for x in range(start[0], start[0] + extent):
                for y in range(start[1], start[1] + extent):
                    neg_pts.append((x,y))            
        self.tot_num_neg = len(neg_pts)
        pos_pts = []
        for x in range(neg_border_width, self.grid_dim + neg_border_width):
            for y in range(neg_border_width, self.grid_dim + neg_border_width):
                point = (x,y)
                if point not in neg_pts:
                    pos_pts.append(point)
                    
        pos = np.array(pos_pts)
        pos = np.insert(pos, 0, 1, axis=1) # Prepend label
        neg = np.array(neg_pts)
        neg = np.insert(neg, 0, 0, axis=1) # Prepend label

        self.grid_dim += 2*neg_border_width
        
        return np.vstack((pos, neg))

    def get_grid_data(self, data):
        grid = np.zeros((self.grid_dim, self.grid_dim))
        for point, value in data:
            point = point.flatten()
            grid[point[1], point[2]] = value
        return grid.T

    def plot_scatter_data(self, data, ax, cname, large=False):
        converter = colors.ColorConverter()
        color = converter.to_rgba(colors.cnames[cname])
        if large:
            self.scatter = ax.scatter(data[1], data[2], color=color, s=200.0, edgecolor='black',
                                      linewidth='3')
        else:
            ax.scatter(data[:,0], data[:,1], color=color, s=25.0)
    
    def plot_gmm(self, samples, gmm, ax, cname):
        preds = gmm.predict(samples)
        means = gmm.means_
        covs  = gmm.covariances_
        color = colors.cnames[cname]
        for i, (mean, cov) in enumerate(zip(means, covs)):
            v, w = np.linalg.eigh(cov)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # Not plotting redundant components
            if not np.any(preds == i):
                continue
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            
    def plot_grid_img(self, data, ax, vmin=0.0, vmax=0.7):
        grid = np.zeros((self.grid_dim, self.grid_dim))
        for point, value in data:
            point = point.flatten()
            grid[point[1], point[2]] = value
        img = ax.imshow(grid.T, origin="lower", vmin=vmin, vmax=vmax, cmap="hot")
        return img

    def plot_line(self, data, ax, lw=5.0, color='g'):
        ax.plot(data, lw=lw, color=color)
        
    def pdf(self, x, mu, sigma):
        try:
            mvn = multivariate_normal(mean=mu, cov=sigma)
            return mvn.pdf(x)
        except np.linalg.LinAlgError as e:
            print "SINGULAR IN TRAJ PROB"
            return 0.0
    
    def gmm_pdf(self, x, gmm, zero_lim=1e-4):
        p = 0.0
        for i in range(gmm.n_components):
            p += gmm.weights_[i] * self.pdf(x, gmm.means_[i], gmm.covariances_[i])
        return max(p, zero_lim)

    def get_probs(self, data):
        pos_probs = []
        neg_probs = []
        for i in range(data.shape[0]):
            x = data[i,1:] # Leave out pre-pended label
            pos_pdf  = self.gmm_pdf(x, self.pos_gmm)
            neg_pdf  = self.gmm_pdf(x, self.neg_gmm)
            pos_prob = pos_pdf / (pos_pdf + neg_pdf)
            neg_prob = neg_pdf / (pos_pdf + neg_pdf)
            pos_probs.append((data[i,:], pos_prob))
            neg_probs.append((data[i,:], neg_prob))
        return pos_probs, neg_probs

    def entropy(self, pos_prob, neg_prob):
        pos_entropy = -(pos_prob * np.log(pos_prob))
        neg_entropy = -(neg_prob * np.log(neg_prob))
        return pos_entropy + neg_entropy

    def get_entropies(self, pos_probs, neg_probs):
        entropies = []
        for i in range(len(pos_probs)):
            e = self.entropy(pos_probs[i][1], neg_probs[i][1])
            entropies.append((pos_probs[i][0], e))
        return entropies


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='foldername', type=str, required=True)
    parser.add_argument('-n', dest='num_frames', type=int, default=5)
    parser.add_argument('-d', dest='grid_dim', type=int, default=40)
    parser.add_argument('-b', dest='neg_border_width', type=int, default=2)
    parser.add_argument('--random', dest='random_sample', action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    neg_starts  = [(5,5), (30,30)]
    neg_extents = [5, 5]
    learner = GMMActiveLearning(args.foldername, args.num_frames, args.grid_dim, neg_starts,
                                neg_extents, True, args.random_sample, args.neg_border_width)
    learner.run()
    learner.progress_bar.close()
