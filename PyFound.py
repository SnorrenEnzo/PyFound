import numpy as np
import numba #for setting dtypes
from numba import jit, njit

from time import time as timer

from scipy.interpolate import griddata
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import erosion, disk

import os
os.environ['NUMBA_DISABLE_PERFORMANCE_WARNINGS'] = '1'

def padMaps(array, x_coords = None, y_coords = None, gridspacing = None):
	"""
	Extrapolate map values to the edges
	"""
	#get top and bottom
	top = array[:1]
	bottom = array[-1:]
	#concatenate
	array = np.concatenate((top, array, bottom), axis = 0)

	#get the sides
	left = array[:,:1]
	right = array[:,-1:]
	#concatenate
	array = np.concatenate((left, array, right), axis = 1)

	#extend the coordinates
	if x_coords is not None and y_coords is not None and gridspacing is not None:
		x_coords = np.concatenate((x_coords[:1] - gridspacing, x_coords, x_coords[-1:] + gridspacing))
		y_coords = np.concatenate((y_coords[:1] - gridspacing, y_coords, y_coords[-1:] + gridspacing))

		return array, x_coords, y_coords
	else:
		return array

def getSky(input, n_iterations = 5, segmap = None, gridspacing = 200):
	"""
	Determine the sky background and RMS
	"""
	data = np.copy(input)

	#make a grid on which the sky background is determined
	#these are the coordinates of the centers of the boxes
	x_coords = np.arange(0, data.shape[0], gridspacing) + gridspacing//2
	y_coords = np.arange(0, data.shape[1], gridspacing) + gridspacing//2


	median_map = np.zeros((len(x_coords), len(y_coords)))
	rms_map = np.zeros((len(x_coords), len(y_coords)))

	#convergence is generally after 5 iterations
	for it in range(n_iterations):
		for i, x in enumerate(x_coords):
			for j, y in enumerate(y_coords):
				gridcell = data[x-gridspacing//2:x+gridspacing//2, y-gridspacing//2:y+gridspacing//2]

				#determine not nan locations
				notnan = ~np.isnan(gridcell)

				#add filtering of segments if a segmap is given
				if segmap is not None:
					filterloc = notnan * (segmap[x-gridspacing//2:x+gridspacing//2, y-gridspacing//2:y+gridspacing//2] == -1)
				else:
					filterloc = notnan

				local_median = np.median(gridcell[filterloc])

				#determine the number of sky pixels; for the first iteration, we
				#assume that all the pixels are sky pixels
				if it == 0:
					Nsky = gridspacing**2
				else:
					#determine the number of non-nan pixels (so number of sky pixels)
					Nsky = np.sum(filterloc)

				sigma_clip = st.norm.ppf(1 - 2/Nsky, loc=0, scale=1)

				#determine standard deviation
				local_rms = np.std(gridcell[filterloc])

				#determine where there are no sky pixels
				not_sky_pix_loc = np.where((gridcell > local_median + local_rms * sigma_clip) + (gridcell < local_median - local_rms * sigma_clip))

				#set the non-sky pixels to nan
				gridcell[not_sky_pix_loc] = np.nan

				median_map[i,j] = local_median
				rms_map[i,j] = local_rms

	#extrapolate the map values to the edges
	median_map, x_coords, y_coords = padMaps(median_map, x_coords, y_coords, gridspacing)
	rms_map = padMaps(rms_map)


	#make a grid
	xv, yv = np.meshgrid(x_coords, y_coords)
	#make the grid on which we will interpolate
	yi, xi = np.indices(data.shape)

	#interpolate the maps to the same shape as the original data
	points = np.array([xv.flatten(), yv.flatten()]).T
	xi = np.array([xi.flatten(), yi.flatten()]).T
	med_map_interp = griddata(points, median_map.flatten(), xi,	method = 'cubic')
	med_map_interp = med_map_interp.reshape(data.shape)
	rms_map_interp = griddata(points, rms_map.flatten(), xi,	method = 'cubic')
	rms_map_interp = rms_map_interp.reshape(data.shape)

	return med_map_interp, rms_map_interp

@njit(cache = True)
def findBrightestPixel(data, sky, sky_RMS, segmap, skycut):
	"""
	Find the location of the brightest pixel above the minimum threshold
	"""
	highvalue = -10000
	highloc = (-1, -1)

	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			if segmap[i][j] == -1\
				and data[i][j] > highvalue\
				and data[i][j] > sky[i][j] + sky_RMS[i][j] * skycut:
				highloc = (i, j)
				highvalue = data[i][j]

	return highloc

@njit(cache = True)
def testFluxValues(i_new, j_new, data, sky_RMS, segmap, segment_it, tolerance, ext):
	"""
	Test the current pixels against the pixel values of both the already segmented
	neighbours and the neighbours not added to the current segment. Return True if:
	1. the current pixel value is lower than all the pixel values of neighbours
		already assigned to the same segment plus a tolerance.
	2. there is no unassigned neighbour which has a higher flux than the current pixel.
	"""
	status = True

	for neigh_i in range(i_new-ext, i_new+ext+1):
		for neigh_j in range(j_new-ext, j_new+ext+1):
			if neigh_i != i_new and neigh_j != j_new\
				and (neigh_i == i_new or neigh_j == j_new): #ensure 4-connection
				#if it is a neighbour already assigned to the segment
				if segmap[neigh_i][neigh_j] == segment_it:
					#test if the pixel value minus a tolerance level is higher than
					#that of the neighbours already assigned to the segment
					if data[i_new][j_new] - sky_RMS[i_new][j_new] * tolerance > data[neigh_i][neigh_j]:
						status = False
				else:
					#if it is not already assigned to the current segment, test
					#if its value is not larger than the current pixel value
					#with a tolerance
					if data[neigh_i][neigh_j] - sky_RMS[neigh_i][neigh_j] * tolerance > data[i_new][j_new]:
						status = False
						# print('no', segment_it, data[neigh_i][neigh_j] - sky_RMS[i_new][j_new] * tolerance, data[i_new][j_new])

	return status

@njit(cache = True)
def addNeighboursToQueue(i, j, data, gradients, segmap, pq_gradients, pq_i, pq_j, min_flux, ext, notassigned_value):
	"""
	Add all neighbours above a specified surface brightness level
	"""
	for neigh_i in range(i-ext, i+ext+1):
		for neigh_j in range(j-ext, j+ext+1):
			if data[neigh_i][neigh_j] >= min_flux and segmap[neigh_i][neigh_j] == notassigned_value:
				pq_gradients.append(gradients[neigh_i][neigh_j])
				pq_i.append(neigh_i)
				pq_j.append(neigh_j)

@njit(cache = True)
def findHighestPriority(pq_gradients, pq_i, pq_j):
	"""
	Find the highest priority object in the queue
	"""
	# highestloc = np.argmax(pq_gradients)

	highestloc = 0
	highestvalue = 0

	for iit in range(len(pq_gradients)):
		if pq_gradients[iit] > highestvalue:
			highestloc = iit

	i_new = pq_i[highestloc]
	j_new = pq_j[highestloc]

	return i_new, j_new, highestloc

@njit(cache = True)
def growSegment(data, sky, sky_RMS, gradients, segmap, start_loc, segment_it, skycut, tolerance, ext):
	"""
	Grow a single segment
	"""

	i, j = start_loc

	#set the segmap value at this location
	segmap[i][j] = segment_it

	notassigned_value = -1

	#initialize the priority queue
	pq_gradients = [gradients[i][j]]
	pq_i = [i]
	pq_j = [j]

	growit = 0
	continue_growing = True
	while continue_growing:
		min_flux = sky[i][j] + sky_RMS[i][j] * skycut

		i_new, j_new, highestloc = findHighestPriority(pq_gradients, pq_i, pq_j)

		#check if pixel value meets criteria
		if testFluxValues(i_new, j_new, data, sky_RMS, segmap, segment_it, tolerance, ext = ext):
			segmap[i_new][j_new] = segment_it

			i = i_new
			j = j_new

			addNeighboursToQueue(i, j, data, gradients, segmap, pq_gradients, pq_i, pq_j, min_flux, ext, notassigned_value)

		#remove this object from the queue
		del(pq_gradients[highestloc])
		del(pq_i[highestloc])
		del(pq_j[highestloc])

		growit += 1

		if len(pq_i) == 0:
			continue_growing = False

def segment(data, sky, sky_RMS, skycut = 4, tolerance = 8, ext = 1):
	"""
	Make a segmentation map
	"""
	#convert to float
	data = data.astype(float)
	sky = sky.astype(float)
	sky_RMS = sky_RMS.astype(float)
	skycut = float(skycut)
	tolerance = float(tolerance)

	#init the segmap
	segmap = np.zeros(data.shape, dtype = int) - 1

	#determine the gradients in the image
	xg, yg = np.gradient(data)
	gradients = (xg**2 + yg**2)**0.5

	#continuously search for new segments until no bright pixel outside all previously
	#determined segments is found
	find_new_segments = True
	segment_it = 0
	starttime = timer()
	while find_new_segments:
		print(f'Segment {segment_it}, runtime = {timer() - starttime:0.03f} s', end = '\r')
		#find the location of the brightest pixel
		brightest_loc = findBrightestPixel(data, sky, sky_RMS, segmap, skycut)

		if brightest_loc == (-1, -1):
			find_new_segments = False
			break

		growSegment(data, sky, sky_RMS, gradients, segmap, brightest_loc, segment_it, skycut, tolerance, ext)

		segment_it += 1

	print('')

	return segmap

@njit(cache = True)
def getSegmentFlux(segment_labels, data, sky, binary_segmap = None, segmap = None):
	"""
	Determine the flux per segment using a given binary segmentation map or normal
	segmentation map.

	binary segmentation map shape: (labels, img_x, img_y)
	segmentation map shape: (img_x, img_y)
	"""
	data_shapex = data.shape[0]
	data_shapey = data.shape[1]

	segment_pixelsum = np.zeros(len(segment_labels))
	segment_area = np.zeros(len(segment_labels))
	if binary_segmap is not None:
		for il in range(len(segment_labels)):
			for i in range(data_shapex):
				for j in range(data_shapey):
					if binary_segmap[il][i][j] > 0:
						segment_pixelsum[il] += data[i][j] - sky[i][j]
						segment_area[il] += 1
	elif segmap is not None:
		for il in range(len(segment_labels)):
			for i in range(data_shapex):
				for j in range(data_shapey):
					if segmap[i][j] == segment_labels[il]:
						segment_pixelsum[il] += data[i][j] - sky[i][j]
						segment_area[il] += 1

	#now determine the flux as the sum of the pixels divided by the number of pixels
	segment_flux = np.zeros(len(segment_labels))
	for il in range(len(segment_labels)):
		if segment_area[il] > 0:
			segment_flux[il] = segment_pixelsum[il] / segment_area[il]

	return segment_flux

@njit(cache = True)
def single_dilate(segmap, input_binary_segmap, segment_labels, data, sky, dilation_kernel):
	"""
	Perform the dilation operations
	"""
	binary_segmap = np.copy(input_binary_segmap)

	#kernel and data shape
	kernel_shapex = dilation_kernel.shape[0]
	kernel_shapey = dilation_kernel.shape[1]
	data_shapex = data.shape[0]
	data_shapey = data.shape[1]

	#loop over all the labels and perform dilation per segment
	for il in range(len(segment_labels)):
		#loop over the image
		for i in range(segmap.shape[0]):
			for j in range(segmap.shape[1]):
				if segmap[i][j] == segment_labels[il]:
					#loop over the dilation kernel
					for a in range(kernel_shapex):
						for b in range(kernel_shapey):
							x = i+a-kernel_shapex//2
							y = j+b-kernel_shapey//2

							if x < data_shapex and y < data_shapey and x >= 0 and y >= 0\
								and dilation_kernel[a][b] > 0:
									binary_segmap[il][x][y] = 1

	#check segment flux
	segment_flux = getSegmentFlux(segment_labels, data, sky, binary_segmap = binary_segmap)

	#now loop over the second and third axes of the binary segmap and check if there is
	#overlap; if so, set all pixels except for the one corresponding to the segment with
	#the highest flux to zero
	for i in range(binary_segmap.shape[1]):
		for j in range(binary_segmap.shape[2]):
			highestflux_labelloc = -1
			for il in range(binary_segmap.shape[0]):
				if binary_segmap[il][i][j] > 0:
					#if we encounter the first label assigned to this pixel
					if highestflux_labelloc == -1:
						highestflux_labelloc = il
					else:
						if segment_flux[il] > segment_flux[highestflux_labelloc]:
							#if the current label has a higher flux than the previous
							#highest label, set that previous segment pixel to 0
							binary_segmap[highestflux_labelloc][i][j] = 0
							highestflux_labelloc = il
						else:
							#if the current segment has a lower flux, set it to zero
							#in the binary segmap
							binary_segmap[il][i][j] = 0

	#check segment flux again
	segment_flux = getSegmentFlux(segment_labels, data, sky, binary_segmap = binary_segmap)

	return binary_segmap, segment_flux

@njit(cache = True)
def binaryToSegmap(binary_segmap, segment_labels):
	"""
	Convert a binary segmap to a normal segmap
	"""
	segmap = np.zeros((binary_segmap.shape[1], binary_segmap.shape[2]), numba.int32) - 1

	for i in range(segmap.shape[0]):
		for j in range(segmap.shape[1]):
			for il in range(len(segment_labels)):
				#check if the segmap is unassinged; this prevents setting the background
				#again and again
				if segmap[i][j] < 0 and binary_segmap[il][i][j] > 0:
					#set to segment
					segmap[i][j] = segment_labels[il]

	return segmap

def perform_dilations(segmap, data, sky, dilation_kernel = disk(4), convergence_threshold = 1.05):
	"""
	Perform iterative dilation until convergence in flux is reached below the threshold
	"""
	dilated_segmap = np.copy(segmap)

	#determine the segment labels
	segment_labels = np.unique(dilated_segmap)
	#remove the background segment label
	segment_labels = np.delete(segment_labels, np.where(segment_labels == -1)[0])

	#arrays for checking convergence
	converged_segments = np.zeros(len(segment_labels), dtype = np.bool)
	previous_converged_segments = np.zeros(len(segment_labels), dtype = np.bool)
	segment_flux = np.zeros(len(segment_labels)) - 1
	binary_segmap = np.zeros((len(segment_labels), dilated_segmap.shape[0], dilated_segmap.shape[1]), dtype = np.int8)

	#make a segmentation map with a different layer for every segment
	binary_segmap = np.zeros((len(segment_labels), dilated_segmap.shape[0], dilated_segmap.shape[1]), dtype = np.int8)

	#dilate 6 times
	starttime = timer()
	for i in range(6):

		new_binary_segmap, new_segment_flux = single_dilate(dilated_segmap, binary_segmap, segment_labels, data, sky, dilation_kernel)

		if i > 0:
			#all segments which had not yet converged in the previous iteration
			#are updated in the binary segmap
			#this prevents updating segments which have already converged
			binary_segmap[~converged_segments] = np.copy(new_binary_segmap[~converged_segments])

			#check convergence per segment
			converged_segments = (new_segment_flux < segment_flux * convergence_threshold)

			#update the segment flux
			segment_flux[~converged_segments] = np.copy(new_segment_flux[~converged_segments])
		else:
			segment_flux = np.copy(new_segment_flux)
			binary_segmap = np.copy(new_binary_segmap)

		dilated_segmap = binaryToSegmap(binary_segmap, segment_labels)

		print(f'Dilation {i}, convergence: {np.sum(converged_segments)}/{len(converged_segments)}, runtime = {timer() - starttime:0.03f} s', end = '\r')

		#if all the segments have converged or no new ones have converged, break
		if np.sum(~converged_segments) == 0 or (np.sum(~previous_converged_segments) - np.sum(~converged_segments) == 0 and i > 0):
			print('\nEarly stopping due to no further convergence')
			break

		previous_converged_segments = converged_segments

	print('')

	return dilated_segmap

def PyFound(data, skycut = 4, tolerance = 16):
	"""
	Extract sources using the ProFound algorithm with each step from the original
	paper by Robotham et al. (2018) page 4 indicated with roman numerals.
	"""
	completestarttime = timer()

	data = data.astype(np.float64)

	#make a smoothed image
	gauss_size = 1
	smooth_data = gaussian_filter(data, gauss_size)

	#### i
	#determine the rough sky background
	sky, sky_RMS = getSky(data)

	#### ii
	#get the segmentation map
	segmap = segment(smooth_data, sky, sky_RMS, skycut = skycut, tolerance = tolerance, ext = 1)


	#### iii
	#now re-calculate the sky background using the segmap
	sky, sky_RMS = getSky(data, segmap = segmap)

	#### iv - viii
	#get the segmentation map
	segmap = segment(smooth_data, sky, sky_RMS, skycut = skycut, tolerance = tolerance, ext = 1)
	#dilate with diameter of 9 pixels iteratively
	segmap = perform_dilations(segmap, data, sky)

	#### ix
	#now re-calculate the sky background using the segmap
	sky, sky_RMS = getSky(data, segmap = segmap)

	print(f'Complete runtime: {timer() - completestarttime:0.03f} s')

	#get edges for visualization
	edges = (segmap - erosion(segmap, disk(1)) > 0).astype(int)

	return segmap, sky, sky_RMS, edges
