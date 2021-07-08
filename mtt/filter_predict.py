class FilterPredict:
    def predict(self, tracks):
        for key, track in tracks:
            if track.stage == 0 or track.stage == 1:
                track.kfilter.predict(measurement=track.measurements[-1]) # could be none but that's cool
                track.predictions.append(track.get_current_guess())
