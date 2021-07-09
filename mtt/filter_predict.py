class FilterPredict:
    def predict(self, tracks=None, measurements=None):
        for key, track in tracks.items():
            if track.stage == 0 or track.stage == 1:
                track.filter_model.predict(measurement=track.measurements[-1]) # could be none but that's cool
                track.predictions.append(track.get_current_guess())
