class FilterPredict:
    def predict(self, tracks=None, measurements=None, time=0):
        for key, track in tracks.items():
            if track.stage == 0 or track.stage == 1:
                track.filter_model.predict(measurement=track.measurements[time]) # arg could be none but that's cool
                #print("current guess: " + str(track.get_current_guess()))
                #track.predictions.append(track.get_current_guess()[0:2])
                track.predictions[time] = track.get_current_guess()[0:2]
