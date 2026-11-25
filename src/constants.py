TENNIS_DATASET_ROOT = "data/tennis"

#: Model input size.  All reported numbers are at this resolution.
HEIGHT = 288
WIDTH = 512

#: Heatmap target parameters produced by ``util.genHeatMap``.
MAG = 1
SIGMA = 2.5

#: Pixel tolerances (Euclidean, input-image pixels) reported by the metrics.
#: AP@4 is the deployment headline; 8 and 12 are loose-localisation context.
TOLERANCES = (4, 8, 12)

#: Reported protocol: train on 3-10, test on 1-2.
TRAIN_GAMES = ("game3", "game4", "game5", "game6",
               "game7", "game8", "game9", "game10")
VAL_GAMES = ("game1", "game2")
