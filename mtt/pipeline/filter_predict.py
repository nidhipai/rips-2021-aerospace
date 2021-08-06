"""
Eduardo Sosa, Tony Zeng, Sal Balkus, Nidhi Pai
Aerospace Team
"""


class FilterPredict:
	"""
	Wrapper around Kalman predict
	"""
	def predict(self, tracks=None, measurements=None, time=0, false_alarms=None):
		"""
		Runs the Kalman filters in each of the tracks
		Args:
			tracks: dictionary of tracks from MTTTracker
			measurements: not used
			time: the current timestep
			false_alarms: not used
		"""
		for key, track in tracks.items():
			if track.stage == 0 or track.stage == 1:
				# measurement could be none but that's cool
				track.filter_model.predict(measurement=track.measurements[time])
				track.predictions[time] = track.get_current_guess()
				#print("CG: (", time, ")", track.get_current_guess())
				track.apriori_pred[time] = track.get_current_apriori_guess()

				mean = (track.filter_model.x_hat[0, 0], track.filter_model.x_hat[1, 0])
				# NOTE: This stores only the position covariance ellipse
				# Here we store the a prior error covariance, which is projected from the previous
				# a posteriori error covariance
				# A smaller a priori error covariance means the actual measurement is trusted less
				# Thus P_minus represents a general area where the filter thinks the next process point will be
				track.apriori_ellipses[time] = [mean, track.filter_model.P_minus]
				track.aposteriori_ellipses[time] = [mean, track.filter_model.P]
				#track.aposteriori_ellipses[time] = [mean, cov] # these are the params needed to make the ellipses
			# don't need to take care of dead objects here beause it's taken care of in get_traj in tracker2
