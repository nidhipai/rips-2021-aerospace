class FilterPredict:
    def predict(self, tracks):
        for track in tracks:
            track.kfilter.predict()
