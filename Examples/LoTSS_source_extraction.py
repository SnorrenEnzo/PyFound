"""
A crude example for source extraction on LoTSS (radio) data
"""

#insert path above to be able to import PyFound
import sys
sys.path.insert(1, '../')

from PyFound import PyFound

import numpy as np
import pandas as pd

from tqdm import tqdm

from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy import units as u

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def wcsTakeFirstTwo(header):
	"""
	Make a new WCS based on the first two axes of a header
	"""
	wcs = WCS(naxis = 2)
	wcs.wcs.ctype = [header['CTYPE1'], header['CTYPE2']]#['RA---TAN', 'DEC--TAN']
	wcs.wcs.crval = [header['CRVAL1'], header['CRVAL2']]
	wcs.wcs.crpix = [header['CRPIX1'], header['CRPIX2']]
	wcs.wcs.cunit = [header['CUNIT1'], header['CUNIT2']]
	wcs.wcs.cdelt = [header['CDELT1'], header['CDELT2']]

	return wcs

def setSubplotParamsWCS(ax):
	ax.set_xlabel('RA')
	ax.set_ylabel('DEC')

def main():
	general_path = '/data2/Mes/lofar_frcnn_tools/mes_development/'

	#location where plots are saved
	plot_saveloc = f'{general_path}Extract_sources/fit_plots_ProFound/'

	datatype = 'LoTSS'

	flux_colorbar_label = 'Flux [mJy/beam]'

	cutout_size = cutout_size = 12 * u.arcmin

	skycut = 3.5
	tolerance = 16

	#load the coordinates; engine = 'python' to prevent warning
	#this is simply a list of interesting compact and extended sources
	df = pd.read_csv(f'{general_path}Make_cutouts/ra_decs.txt', sep = ', ', engine = 'python')

	#get coordinates
	RA = np.array(df['RA'], dtype = float)
	DEC = np.array(df['DEC'], dtype = float)


	for i in tqdm(range(61, 91)):
		fieldname = df['Field_name'][i]
		fits_fname = f'{general_path}DR2_mosaics/{fieldname}.fits'

		#make astropy sky coordinates
		cutout_coords = SkyCoord(RA[i], DEC[i], frame = 'icrs', unit = (u.deg, u.deg))

		#first load the image
		with fits.open(fits_fname, memmap = True) as hdul:
			img_data = hdul[0].data
			img_header = hdul[0].header

			wcs = WCS(img_header)

		#if we have two axes too many, then remove these and fix the wcs
		if len(img_data.shape) == 4:
			img_data = img_data[0,0]
			wcs = wcsTakeFirstTwo(img_header)

		cutout = Cutout2D(img_data, cutout_coords, cutout_size, wcs = wcs, mode = 'partial')
		cutout_data = cutout.data
		cutout_wcs = cutout.wcs

		#convert to mJy
		if datatype == 'LoTSS':
			cutout_data = cutout_data * 1e3

		#### now start the extraction
		segmap, sky, sky_RMS, edges = PyFound(cutout_data, skycut = skycut, tolerance = tolerance)
		labels = np.unique(segmap)

		fig = plt.figure(figsize = (14, 11))

		#### Plot the flux and the segmentation edges
		ax1 = fig.add_subplot(221, projection = cutout_wcs)

		#make a power law normalization
		norm = mcolors.PowerNorm(gamma = 0.6)
		imagerange_RMS = np.median(sky_RMS)
		#plot the flux
		im1 = ax1.imshow(cutout_data, origin = 'lower', cmap = 'Blues',
					vmin = -2*imagerange_RMS, vmax = 20*imagerange_RMS, norm = norm)
		#add segment edges
		cmap_reds = plt.get_cmap('Reds').reversed()
		edges_rgb = cmap_reds(edges)
		edges_rgb[edges == 0] = (0, 0, 0, 0)
		edges_rgb[edges != 0] = (1, 0, 0, 1)
		im11 = ax1.imshow(edges_rgb)
		ax1.set_title('Flux')
		cbar = plt.colorbar(im1, ax = ax1)
		cax = cbar.ax.set_ylabel(flux_colorbar_label)
		setSubplotParamsWCS(ax1)

		#### Plot the segmentation map
		ax2 = fig.add_subplot(222, sharex = ax1, sharey = ax1, projection = cutout_wcs)
		#plot the segmentation map with no label having the color white
		cmap_rainbow = plt.get_cmap('jet')
		#redefine the value for the background (-1)
		cmap_rainbow.set_under('w')
		im2 = ax2.imshow(segmap, cmap = cmap_rainbow, vmin = -0.001)
		cbar = fig.colorbar(im2, ax = ax2)
		cax = cbar.ax.set_ylabel('Segmentation labels')
		ax2.set_title('Segmentation map')
		setSubplotParamsWCS(ax2)

		#### Plot the sky background map
		ax3 = fig.add_subplot(223, sharex = ax1, sharey = ax1, projection = cutout_wcs)
		im3 = ax3.imshow(sky, cmap = 'Blues')
		cbar = fig.colorbar(im3, ax = ax3)
		cax = cbar.ax.set_ylabel(flux_colorbar_label)
		ax3.set_title('Sky background')
		setSubplotParamsWCS(ax3)

		#### Plot the sky RMS map
		ax4 = fig.add_subplot(224, sharex = ax1, sharey = ax1, projection = cutout_wcs)
		im4 = ax4.imshow(sky_RMS, cmap = 'Blues')
		cbar = fig.colorbar(im4, ax = ax4)
		cax = cbar.ax.set_ylabel(flux_colorbar_label)
		ax4.set_title('Sky RMS')
		setSubplotParamsWCS(ax4)

		# ax1.set_xlim((740, 780))
		# ax1.set_ylim((560, 520))

		fig.subplots_adjust(wspace = 0.15)

		plt.savefig(f'{plot_saveloc}{datatype}_{i}_{fieldname}_ProFound_extraction_grid=200_minf={skycut}_tol={tolerance}.png', dpi = 300, bbox_inches = 'tight')
		# plt.show()
		plt.close()


if __name__ == '__main__':
	main()
